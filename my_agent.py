import os
import json
import logging
import requests
from typing import TypedDict, Annotated

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime import BedrockAgentCoreContext
from bedrock_agentcore.services.identity import IdentityClient
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pathlib import Path

app = BedrockAgentCoreApp()

logger = logging.getLogger("gh-agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

CALLBACK_URL = os.environ.get(
    "CALLBACK_URL",
    "https://legendary-memory-4j45x49gp6p63j5gr-8501.app.github.dev/?auth=success",
)


class AuthRequiredException(BaseException):
    def __init__(self, url):
        self.url = url
        super().__init__(url)


def _get_workload_token():
    """Get workload access token — check runtime context first, then fall back to local dev."""
    # Try runtime context (when running in AgentCore)
    token = BedrockAgentCoreContext.get_workload_access_token()
    if token:
        logger.info("Got workload token from runtime context")
        return token

    # Local dev fallback — read from .agentcore.json
    logger.info("No runtime context token, using local dev fallback")
    config_path = Path(".agentcore.json")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f) or {}

    workload_name = config.get("workload_identity_name", "myagent-workload")
    user_id = config.get("user_id", "agent001")
    logger.info(f"Using workload={workload_name}, user_id={user_id}")

    client = IdentityClient()
    resp = client.get_workload_access_token(workload_name, user_id=user_id)
    return resp["workloadAccessToken"]


def _get_github_token():
    """
    Call the identity API directly to get a GitHub OAuth token.
    Returns the access_token string, or raises AuthRequiredException with the auth URL.
    """
    client = IdentityClient()
    workload_token = _get_workload_token()

    # Call get_resource_oauth2_token directly
    req = {
        "resourceCredentialProviderName": "github-provider",
        "scopes": ["repo", "read:user"],
        "oauth2Flow": "USER_FEDERATION",
        "workloadIdentityToken": workload_token,
        "resourceOauth2ReturnUrl": CALLBACK_URL,
    }

    logger.info("Calling get_resource_oauth2_token...")
    response = client.dp_client.get_resource_oauth2_token(**req)

    # Log what we got back
    safe_resp = {k: v for k, v in response.items() if k != "accessToken"}
    logger.info(f"Response keys: {list(response.keys())}")
    logger.info(f"Response (sans token): {safe_resp}")

    # If we got a token, return it
    if "accessToken" in response:
        logger.info("Got access token!")
        return response["accessToken"]

    # If we got an auth URL, return it to the user
    if "authorizationUrl" in response:
        auth_url = response["authorizationUrl"]
        logger.info(f"Got authorization URL: {auth_url[:80]}...")
        raise AuthRequiredException(auth_url)

    raise RuntimeError(f"Unexpected response: {list(response.keys())}")


def _fetch_github_data():
    """Fetch GitHub data using the OAuth token from identity."""
    token = _get_github_token()
    headers = {"Authorization": f"Bearer {token}"}

    profile_resp = requests.get("https://api.github.com/user", headers=headers)
    if profile_resp.status_code != 200:
        return f"Error fetching profile: {profile_resp.text}"
    profile = profile_resp.json()

    repos_resp = requests.get(
        "https://api.github.com/user/repos",
        headers=headers,
        params={"sort": "pushed", "per_page": 30, "type": "owner"},
    )
    repos = repos_resp.json() if repos_resp.status_code == 200 else []

    repo_summaries = []
    for r in repos[:10]:
        lang = r.get("language") or "N/A"
        stars = r.get("stargazers_count", 0)
        desc = r.get("description") or "No description"
        repo_summaries.append(f"  - {r['name']} ({lang}, {stars} stars): {desc}")

    lang_count: dict[str, int] = {}
    for r in repos:
        lang = r.get("language")
        if lang:
            lang_count[lang] = lang_count.get(lang, 0) + 1
    top_langs = sorted(lang_count.items(), key=lambda x: x[1], reverse=True)[:5]
    lang_str = ", ".join(f"{l} ({c})" for l, c in top_langs) if top_langs else "N/A"

    return (
        f"GitHub Profile: {profile.get('login')} ({profile.get('name', 'N/A')})\n"
        f"Bio: {profile.get('bio') or 'N/A'}\n"
        f"Followers: {profile.get('followers')} | Following: {profile.get('following')}\n"
        f"Public Repos: {profile.get('public_repos')}\n"
        f"Top Languages: {lang_str}\n\n"
        f"Recently Active Repos:\n" + "\n".join(repo_summaries)
    )


@tool
def fetch_github_profile():
    """Fetches the authenticated user's GitHub profile, repos, and language stats."""
    return _fetch_github_data()


# ── LangGraph setup ──
tools = [fetch_github_profile]
llm = ChatBedrockConverse(
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
)
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

    try:
        user_message = payload.get("prompt", "Hello")
        res = graph.invoke({"messages": [HumanMessage(content=user_message)]})
        return {"result": res["messages"][-1].content}

    except AuthRequiredException as e:
        return {"result": f"__AUTH_REQUIRED__{e.url}"}

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"result": f"Error: {str(e)}"}


if __name__ == "__main__":
    app.run()