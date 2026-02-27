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

CALLBACK_URL = os.environ.get(
    "CALLBACK_URL",
    "https://gh-agent-aci-aq6nrmcvn96tycbvhnqjgf.streamlit.app",
)


class AuthRequiredException(Exception):
    """Raised when the user needs to authorize via OAuth."""
    def __init__(self, message):
        self.message = message
        super().__init__(message)


def _handle_auth_url(url: str):
    """
    Official callback for the @requires_access_token decorator.
    Called with the authorization URL — we raise immediately to stop polling
    and return the URL to the user.
    """
    raise AuthRequiredException(
        f"🔒 **GitHub Authorization Required**\n\n"
        f"[Click here to authorize GitHub access]({url})\n\n"
        f"After authorizing, come back and retry your command."
    )


@requires_access_token(
    provider_name="github-provider",
    scopes=["repo", "read:user"],
    auth_flow="USER_FEDERATION",
    callback_url=CALLBACK_URL,
    on_auth_url=_handle_auth_url,
)
def _get_github_data(access_token=None):
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


@tool
def fetch_github_profile():
    """Fetches the authenticated user's GitHub profile, repos, and language stats."""
    return _get_github_data()


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
