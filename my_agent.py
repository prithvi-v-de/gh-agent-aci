import os
import json
import re
import logging
import threading
import queue
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

CALLBACK_URL = os.environ.get(
    "CALLBACK_URL",
    "https://gh-agent-aci-aq6nrmcvn96tycbvhnqjgf.streamlit.app",
)


# ── Capture auth URLs from the SDK's log/print output ──
_auth_url_store = []


class _AuthURLLogHandler(logging.Handler):
    """Intercepts log messages to capture the OAuth authorization URL."""
    def emit(self, record):
        msg = record.getMessage()
        if "authorization url" in msg.lower() or "authorize?" in msg.lower():
            match = re.search(r"(https://[^\s'\"]+)", msg)
            if match:
                _auth_url_store.append(match.group(1))


# Attach to root logger so we catch the SDK's polling messages
_handler = _AuthURLLogHandler()
_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_handler)
logging.getLogger().setLevel(logging.DEBUG)


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
        repo_summaries.append(
            f"  - **{r['name']}** ({lang}, {stars} stars): {desc}"
        )

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


def _call_github_with_timeout(timeout_seconds=15):
    """
    Call _get_github_data in a thread. If the OAuth decorator starts polling
    (because user hasn't authorized yet), we capture the auth URL from logs
    and return it instead of blocking forever.
    """
    _auth_url_store.clear()
    result_queue = queue.Queue()

    def _run():
        try:
            data = _get_github_data()
            result_queue.put(("ok", data))
        except Exception as e:
            result_queue.put(("error", e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    # If the thread finished, return its result
    if not result_queue.empty():
        status, value = result_queue.get_nowait()
        if status == "ok":
            return value
        else:
            raise value

    # Thread is still running (blocked in polling) — check for captured auth URL
    if _auth_url_store:
        url = _auth_url_store[0]
        raise AuthRequiredException(
            f"🔒 **GitHub Authorization Required**\n\n"
            f"Please click the link below to authorize access, then retry your command:\n\n"
            f"[Authorize GitHub Access]({url})"
        )

    raise AuthRequiredException(
        "⚠️ **Timeout:** The agent is waiting for authorization but could not determine the auth URL. "
        "Please check your AgentCore Identity configuration."
    )


@tool
def fetch_github_profile():
    """Fetches the authenticated user's GitHub profile, repos, and language stats. Requires no arguments."""
    return _call_github_with_timeout(timeout_seconds=15)


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

    except AuthRequiredException as auth_err:
        return {"result": auth_err.message}

    except Exception as e:
        return {"result": f"⚠️ Backend Error: {str(e)}"}


if __name__ == "__main__":
    app.run()