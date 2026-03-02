import os
import json
import asyncio
import logging
import requests
from typing import TypedDict, Annotated

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.identity.auth import requires_access_token
from bedrock_agentcore.services.identity import TokenPoller
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

# ── Shared state ──
_auth_state = {"url": None}
_token_cache = {"token": None}


class AuthRequiredException(BaseException):
    """BaseException so it escapes everything."""
    def __init__(self, url):
        self.url = url
        super().__init__(url)


def _store_auth_url(url: str):
    """on_auth_url callback — stores URL, does NOT raise."""
    logger.info(f"on_auth_url received: {url[:80]}...")
    _auth_state["url"] = url


class _QuickTokenPoller(TokenPoller):
    """Waits 1s then raises with auth URL instead of polling for 600s."""
    async def poll_for_token(self) -> str:
        await asyncio.sleep(1)
        url = _auth_state.get("url", "")
        if url:
            raise AuthRequiredException(url)
        raise RuntimeError("No auth URL captured and no token available")


# ── This function uses the decorator to get the GitHub token ──
# The decorator hooks into the app's identity system which has the
# correct runtime-provided workload token (consistent across calls).
@requires_access_token(
    provider_name="github-provider",
    scopes=["repo", "read:user"],
    auth_flow="USER_FEDERATION",
    callback_url=CALLBACK_URL,
    on_auth_url=_store_auth_url,
    token_poller=_QuickTokenPoller(),
)
def _get_github_token(access_token=None):
    """Called OUTSIDE LangGraph. Returns the token or raises AuthRequiredException."""
    logger.info("Got GitHub access token from decorator!")
    return access_token


def _fetch_github_data(token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 1. Fetch Core Profile Data
    profile_resp = requests.get("https://api.github.com/user", headers=headers)
    if profile_resp.status_code != 200:
        return f"Error fetching profile: {profile_resp.text}"
    profile = profile_resp.json()

    # 2. Fetch Up to 100 Repositories for Better Analytics
    repos_resp = requests.get(
        "https://api.github.com/user/repos",
        headers=headers,
        params={"sort": "updated", "per_page": 100, "type": "owner"},
    )
    repos = repos_resp.json() if repos_resp.status_code == 200 else []

    # 3. Calculate Deep Metrics (The "Portfolio" Stats)
    total_stars = sum(r.get("stargazers_count", 0) for r in repos)
    total_issues = sum(r.get("open_issues_count", 0) for r in repos)
    
    # Calculate Dominant Languages
    lang_count: dict[str, int] = {}
    for r in repos:
        lang = r.get("language")
        if lang:
            lang_count[lang] = lang_count.get(lang, 0) + 1
    
    # Grab the top 3 languages
    top_langs = sorted(lang_count.items(), key=lambda x: x[1], reverse=True)[:3]
    lang_str = ", ".join(f"{l} ({c} repos)" for l, c in top_langs) if top_langs else "N/A"

    # 4. Isolate the "Star" Projects (Sort by stars instead of just recent pushes)
    top_repos = sorted(repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)[:5]
    repo_summaries = []
    for r in top_repos:
        lang = r.get("language") or "N/A"
        stars = r.get("stargazers_count", 0)
        desc = r.get("description") or "No description"
        repo_summaries.append(f"  - {r['name']} ({lang}, {stars} ⭐): {desc}")

    # 5. Format a Data-Rich String for Claude
    created_at = profile.get('created_at', '')[:10] if profile.get('created_at') else 'Unknown'
    
    return (
        f"👤 GitHub Profile: {profile.get('login')} ({profile.get('name', 'N/A')})\n"
        f"📅 Account Created: {created_at}\n"
        f"🏢 Company: {profile.get('company') or 'N/A'} | 📍 Location: {profile.get('location') or 'N/A'}\n"
        f"📖 Bio: {profile.get('bio') or 'N/A'}\n"
        f"👥 Followers: {profile.get('followers')} | Following: {profile.get('following')}\n\n"
        f"📊 DEEP METRICS (Based on {len(repos)} recent repos):\n"
        f"  - Total Stars Earned: {total_stars}\n"
        f"  - Total Open Issues Managed: {total_issues}\n"
        f"  - Dominant Stack: {lang_str}\n\n"
        f"🌟 Top 5 Highlighted Repositories:\n" + "\n".join(repo_summaries)
    )

# ── LangGraph tool uses the pre-fetched cached token ──
@tool
def fetch_github_profile():
    """Fetches the authenticated user's GitHub profile, repos, and language stats."""
    token = _token_cache.get("token")
    if not token:
        return "Error: No GitHub token available. Please authorize first."
    return _fetch_github_data(token)


# ── LangGraph setup ──
tools = [fetch_github_profile]
llm = ChatBedrockConverse(
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-2",
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

    # Reset state each invocation
    _auth_state["url"] = None
    _token_cache["token"] = None

    # ── STEP 1: Get GitHub token OUTSIDE LangGraph ──
    # The decorator uses the runtime's workload token (consistent across calls).
    # We catch the auth exception HERE instead of letting LangGraph swallow it.
    try:
        token = _get_github_token()
        _token_cache["token"] = token
        logger.info("GitHub token obtained, proceeding to LangGraph")
    except AuthRequiredException as e:
        logger.info(f"Auth required, returning URL to frontend")
        return {"result": f"__AUTH_REQUIRED__{e.url}"}
    except Exception as e:
        logger.error(f"Auth check failed: {e}", exc_info=True)
        if _auth_state.get("url"):
            return {"result": f"__AUTH_REQUIRED__{_auth_state['url']}"}
        return {"result": f"Error during auth check: {str(e)}"}

    # ── STEP 2: Run LangGraph with pre-fetched token ──
    try:
        user_message = payload.get("prompt", "Hello")
        res = graph.invoke({"messages": [HumanMessage(content=user_message)]})
        return {"result": res["messages"][-1].content}
    except Exception as e:
        logger.error(f"Graph error: {e}", exc_info=True)
        return {"result": f"Error: {str(e)}"}


if __name__ == "__main__":
    app.run()
#Force Id Sync
