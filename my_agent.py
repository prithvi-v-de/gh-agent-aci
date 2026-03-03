import os

os.environ["AWS_DEFAULT_REGION"] = "us-east-2"
os.environ["AWS_REGION"] = "us-east-2"

import json
import time
import logging
import requests
from typing import TypedDict, Annotated

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.services.identity import IdentityClient
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

app = BedrockAgentCoreApp()

logger = logging.getLogger("gh-agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

CALLBACK_URL = os.environ.get(
    "CALLBACK_URL",
    "https://mmwr8fz3ux.us-east-2.awsapprunner.com/?auth=success",
)

_token_cache = {"token": None}


def _get_workload_token():
    from pathlib import Path
    config_path = Path("/var/task/.agentcore.json")
    if not config_path.exists():
        config_path = Path(".agentcore.json")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f) or {}
    workload_name = config.get("workload_identity_name", "gh-agent-prod")
    user_id = config.get("user_id", "agent001")
    logger.info(f"Using workload={workload_name}, user_id={user_id}")
    identity = IdentityClient(region="us-east-2")
    resp = identity.dp_client.get_workload_access_token_for_user_id(
        workloadName=workload_name, userId=user_id,
    )
    return resp["workloadAccessToken"]


def _get_github_token(session_uri=None):
    workload_token = _get_workload_token()
    identity = IdentityClient(region="us-east-2")
    req = {
        "resourceCredentialProviderName": "github-provider",
        "scopes": ["repo", "read:user"],
        "oauth2Flow": "USER_FEDERATION",
        "workloadIdentityToken": workload_token,
        "resourceOauth2ReturnUrl": CALLBACK_URL,
    }
    if session_uri:
        req["sessionUri"] = session_uri
        logger.info("Including sessionUri from previous auth request")
    logger.info(f"Calling get_resource_oauth2_token (has_sessionUri={'yes' if session_uri else 'no'})...")
    
    try:
        response = identity.dp_client.get_resource_oauth2_token(**req)
    except Exception as e:
        # If sessionUri is invalid/expired, retry without it
        if "Invalid sessionUri" in str(e) and session_uri:
            logger.warning("sessionUri invalid, retrying without it")
            del req["sessionUri"]
            response = identity.dp_client.get_resource_oauth2_token(**req)
        else:
            raise
    logger.info(f"Response keys: {list(response.keys())}")

    if "accessToken" in response:
        logger.info("SUCCESS: Got access token!")
        return response["accessToken"], None

    if "authorizationUrl" in response:
        auth_url = response["authorizationUrl"]
        new_session_uri = response.get("sessionUri", "")
        logger.info(f"Auth needed. sessionUri present: {bool(new_session_uri)}")
        return None, {"auth_url": auth_url, "session_uri": new_session_uri}

    # sessionStatus means token is still being processed — poll a few times
    if "sessionStatus" in response:
        status = response.get("sessionStatus", "")
        logger.info(f"Session status: {status} — polling for token...")
        for attempt in range(10):
            time.sleep(3)
            retry_resp = identity.dp_client.get_resource_oauth2_token(**req)
            logger.info(f"Poll {attempt+1}: keys={list(retry_resp.keys())}")
            if "accessToken" in retry_resp:
                logger.info("SUCCESS: Got access token after polling!")
                return retry_resp["accessToken"], None
            if "authorizationUrl" in retry_resp:
                auth_url = retry_resp["authorizationUrl"]
                new_session_uri = retry_resp.get("sessionUri", "")
                return None, {"auth_url": auth_url, "session_uri": new_session_uri}
        # All polls exhausted — return pending status
        logger.warning("Token not ready after 10 polls")
        return None, {"status": "pending", "session_uri": session_uri or ""}

    raise RuntimeError(f"Unexpected response: {list(response.keys())}")


def _fetch_github_data(token):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    profile_resp = requests.get("https://api.github.com/user", headers=headers)
    if profile_resp.status_code != 200:
        return f"Error fetching profile: {profile_resp.text}"
    profile = profile_resp.json()
    repos_resp = requests.get("https://api.github.com/user/repos", headers=headers,
        params={"sort": "updated", "per_page": 100, "type": "owner"})
    repos = repos_resp.json() if repos_resp.status_code == 200 else []
    total_stars = sum(r.get("stargazers_count", 0) for r in repos)
    total_issues = sum(r.get("open_issues_count", 0) for r in repos)
    lang_count = {}
    for r in repos:
        lang = r.get("language")
        if lang:
            lang_count[lang] = lang_count.get(lang, 0) + 1
    top_langs = sorted(lang_count.items(), key=lambda x: x[1], reverse=True)[:3]
    lang_str = ", ".join(f"{l} ({c} repos)" for l, c in top_langs) if top_langs else "N/A"
    top_repos = sorted(repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)[:5]
    repo_summaries = []
    for r in top_repos:
        lang = r.get("language") or "N/A"
        stars = r.get("stargazers_count", 0)
        desc = r.get("description") or "No description"
        repo_summaries.append(f"  - {r['name']} ({lang}, {stars} stars): {desc}")
    created_at = profile.get("created_at", "")[:10] if profile.get("created_at") else "Unknown"
    return (
        f"GitHub Profile: {profile.get('login')} ({profile.get('name', 'N/A')})\n"
        f"Account Created: {created_at}\n"
        f"Company: {profile.get('company') or 'N/A'} | Location: {profile.get('location') or 'N/A'}\n"
        f"Bio: {profile.get('bio') or 'N/A'}\n"
        f"Followers: {profile.get('followers')} | Following: {profile.get('following')}\n\n"
        f"DEEP METRICS (Based on {len(repos)} recent repos):\n"
        f"  - Total Stars Earned: {total_stars}\n"
        f"  - Total Open Issues Managed: {total_issues}\n"
        f"  - Dominant Stack: {lang_str}\n\n"
        f"Top 5 Highlighted Repositories:\n" + "\n".join(repo_summaries)
    )


@tool
def fetch_github_profile():
    """Fetches the authenticated user's GitHub profile, repos, and language stats."""
    token = _token_cache.get("token")
    if not token:
        return "Error: No GitHub token available. Please authorize first."
    return _fetch_github_data(token)


tools = [fetch_github_profile]
llm = ChatBedrockConverse(model="us.anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-east-2")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot_node(state: State) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
graph = builder.compile()


@app.entrypoint
def invoke(payload):
    if payload.get("type") == "warmup":
        return {"result": "Agent is warm and ready."}
    _token_cache["token"] = None
    session_uri = payload.get("session_uri")
    try:
        token, auth_info = _get_github_token(session_uri=session_uri)
        if token:
            _token_cache["token"] = token
            logger.info("GitHub token obtained, proceeding to LangGraph")
        else:
            # Check if it's a pending status (auth done, token not ready yet)
            if auth_info.get("status") == "pending":
                logger.info("Token pending, returning retry signal")
                return {
                    "result": "__TOKEN_PENDING__",
                    "session_uri": auth_info["session_uri"],
                }
            logger.info("Auth required, returning URL + sessionUri")
            return {
                "result": f"__AUTH_REQUIRED__{auth_info['auth_url']}",
                "session_uri": auth_info["session_uri"],
            }
    except Exception as e:
        logger.error(f"Auth check failed: {e}", exc_info=True)
        return {"result": f"Error during auth check: {str(e)}"}
    try:
        user_message = payload.get("prompt", "Hello")
        res = graph.invoke({"messages": [HumanMessage(content=user_message)]})
        return {"result": res["messages"][-1].content}
    except Exception as e:
        logger.error(f"Graph error: {e}", exc_info=True)
        return {"result": f"Error: {str(e)}"}

if __name__ == "__main__":
    app.run()
