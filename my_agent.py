import os
import json
import re
import requests
from typing import TypedDict, Annotated

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.identity.auth import requires_access_token
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

app = BedrockAgentCoreApp()

# ── Streamlit callback URL (where users return after GitHub OAuth) ──
CALLBACK_URL = os.environ.get(
    "CALLBACK_URL",
    "https://gh-agent-aci-aq6nrmcvn96tycbvhnqjgf.streamlit.app",
)


# ── Custom exception for the OAuth redirect flow ──
class AuthRequiredException(BaseException):
    def __init__(self, message):
        self.message = message


# ── GitHub data fetcher (OAuth-protected) ──
@requires_access_token(
    provider_name="github-provider",
    scopes=["repo", "read:user"],
    auth_flow="USER_FEDERATION",
    callback_url=CALLBACK_URL,
)
def _get_github_data(access_token=None):
    """Fetch the authenticated user's profile, repos, and top-language stats."""
    headers = {"Authorization": f"Bearer {access_token}"}

    # 1. Basic profile
    profile_resp = requests.get("https://api.github.com/user", headers=headers)
    if profile_resp.status_code != 200:
        return f"Error fetching profile: {profile_resp.text}"
    profile = profile_resp.json()

    # 2. Public repos (up to 30, sorted by most recently pushed)
    repos_resp = requests.get(
        "https://api.github.com/user/repos",
        headers=headers,
        params={"sort": "pushed", "per_page": 30, "type": "owner"},
    )
    repos = repos_resp.json() if repos_resp.status_code == 200 else []

    # 3. Build a summary with languages
    repo_summaries = []
    for r in repos[:10]:  # top 10 most-recently-pushed
        lang = r.get("language") or "N/A"
        stars = r.get("stargazers_count", 0)
        desc = r.get("description") or "No description"
        repo_summaries.append(
            f"  • **{r['name']}** ({lang}, ⭐ {stars}): {desc}"
        )

    # 4. Aggregate language breakdown
    lang_count: dict[str, int] = {}
    for r in repos:
        lang = r.get("language")
        if lang:
            lang_count[lang] = lang_count.get(lang, 0) + 1
    top_langs = sorted(lang_count.items(), key=lambda x: x[1], reverse=True)[:5]
    lang_str = ", ".join(f"{l} ({c})" for l, c in top_langs) if top_langs else "N/A"

    return (
        f"**GitHub Profile:** {profile.get('login')} ({profile.get('name', 'N/A')})\n"
        f"**Bio:** {profile.get('bio') or 'N/A'}\n"
        f"**Followers:** {profile.get('followers')} | **Following:** {profile.get('following')}\n"
        f"**Public Repos:** {profile.get('public_repos')}\n"
        f"**Top Languages:** {lang_str}\n\n"
        f"**Recently Active Repos:**\n" + "\n".join(repo_summaries)
    )


# ── Tool definition ──
@tool
def fetch_github_profile():
    """Fetches the authenticated user's GitHub profile, repos, and language stats. Requires no arguments."""
    try:
        return _get_github_data()
    except AuthRequiredException:
        raise
    except Exception as e:
        err_str = str(e)
        # Try to extract the OAuth authorization URL from the exception
        url = getattr(e, "authorization_url", None) or getattr(e, "url", None) or getattr(e, "redirect_url", None)
        if not url:
            match = re.search(r"(https://[^\s'\">\]]+)", err_str)
            if match:
                url = match.group(1)
        if url:
            raise AuthRequiredException(
                f"🔒 **Permission Required:** Please [Authorize GitHub Access]({url}) and then retry your command."
            )
        raise AuthRequiredException(
            f"⚠️ **Authorization Failed:** Could not extract auth link.\nRaw error: {err_str}"
        )


# ── LangGraph setup (compiled once at module level → no cold-start penalty) ──
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


# ── Agent entrypoint ──
@app.entrypoint
def invoke(payload):
    # Fast warmup path
    if payload.get("type") == "warmup":
        return {"result": "Agent is warm and ready."}

    try:
        user_message = payload.get("prompt", "Hello")
        res = graph.invoke({"messages": [HumanMessage(content=user_message)]})
        return {"result": res["messages"][-1].content}

    except AuthRequiredException as auth_err:
        return {"result": auth_err.message}

    except Exception as e:
        return {"result": f"⚠️ Backend Error: {str(e)}"}


if __name__ == "__main__":
    app.run()
