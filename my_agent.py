import os
import sys
import io
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


class AuthRequiredException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


# ── Thread-safe list to collect auth URLs from BOTH stdout and logging ──
_captured_urls = []
_capture_lock = threading.Lock()


def _add_url(url):
    with _capture_lock:
        if url not in _captured_urls:
            _captured_urls.append(url)


def _get_urls():
    with _capture_lock:
        return list(_captured_urls)


def _clear_urls():
    with _capture_lock:
        _captured_urls.clear()


# ── Logging handler to catch auth URLs from the logging module ──
class _AuthLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            url = _extract_auth_url(msg)
            if url:
                _add_url(url)
        except Exception:
            pass


# Install on ALL loggers (root + any SDK-specific ones)
_log_handler = _AuthLogHandler()
_log_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_log_handler)
logging.getLogger().setLevel(logging.DEBUG)
# Also try common SDK logger names
for logger_name in ["bedrock_agentcore", "bedrock_agentcore.identity", "botocore", "urllib3"]:
    logging.getLogger(logger_name).addHandler(_log_handler)
    logging.getLogger(logger_name).setLevel(logging.DEBUG)


# ── Stdout/stderr interceptor that also scans for URLs ──
class _URLCapturingWriter:
    """Wraps a StringIO and scans every write for auth URLs."""
    def __init__(self, buffer):
        self.buffer = buffer

    def write(self, text):
        self.buffer.write(text)
        url = _extract_auth_url(text)
        if url:
            _add_url(url)
        return len(text)

    def flush(self):
        self.buffer.flush()

    def getvalue(self):
        return self.buffer.getvalue()


@requires_access_token(
    provider_name="github-provider",
    scopes=["repo", "read:user"],
    auth_flow="USER_FEDERATION",
    callback_url=CALLBACK_URL,
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


def _extract_auth_url(text):
    """Extract an authorization URL from text."""
    if not text:
        return None
    # AgentCore authorize URL
    match = re.search(r"(https://bedrock-agentcore[^\s'\"]+)", text)
    if match:
        return match.group(1)
    # Any URL with authorize/oauth
    match = re.search(r"(https://[^\s'\"]*(?:authorize|oauth)[^\s'\"]*)", text)
    if match:
        return match.group(1)
    return None


def _call_github_with_timeout(timeout_seconds=25):
    _clear_urls()
    result_queue = queue.Queue()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    def _run():
        old_stdout, old_stderr = sys.stdout, sys.stderr
        # Wrap in URL-scanning writers
        sys.stdout = _URLCapturingWriter(stdout_buf)
        sys.stderr = _URLCapturingWriter(stderr_buf)
        try:
            data = _get_github_data()
            result_queue.put(("ok", data))
        except Exception as e:
            result_queue.put(("error", e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    elapsed = 0
    while elapsed < timeout_seconds:
        t.join(timeout=2)

        # Thread finished?
        if not result_queue.empty():
            status, value = result_queue.get_nowait()
            if status == "ok":
                return value
            err_str = str(value)
            url = _extract_auth_url(err_str)
            if url:
                raise AuthRequiredException(
                    f"🔒 **GitHub Authorization Required**\n\n"
                    f"[Click here to authorize GitHub access]({url})\n\n"
                    f"After authorizing, come back and retry your command."
                )
            raise value

        # Check both sources for auth URL
        urls = _get_urls()
        if urls:
            raise AuthRequiredException(
                f"🔒 **GitHub Authorization Required**\n\n"
                f"[Click here to authorize GitHub access]({urls[0]})\n\n"
                f"After authorizing, come back and retry your command."
            )

        elapsed += 2

    # Final check
    urls = _get_urls()
    if urls:
        raise AuthRequiredException(
            f"🔒 **GitHub Authorization Required**\n\n"
            f"[Click here to authorize GitHub access]({urls[0]})\n\n"
            f"After authorizing, come back and retry your command."
        )

    all_output = stdout_buf.getvalue() + stderr_buf.getvalue()
    raise AuthRequiredException(
        f"⚠️ **Timed out.** Could not find auth URL in stdout, stderr, or logs.\n\n"
        f"stdout: ```{stdout_buf.getvalue()[:300] or 'EMPTY'}```\n\n"
        f"stderr: ```{stderr_buf.getvalue()[:300] or 'EMPTY'}```\n\n"
        f"captured_urls: {_get_urls()}"
    )


@tool
def fetch_github_profile():
    """Fetches the authenticated user's GitHub profile, repos, and language stats."""
    try:
        return _call_github_with_timeout(timeout_seconds=25)
    except AuthRequiredException:
        raise
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


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
        err_str = str(e)
        url = _extract_auth_url(err_str)
        if url:
            return {
                "result": f"🔒 **GitHub Authorization Required**\n\n"
                          f"[Click here to authorize GitHub access]({url})\n\n"
                          f"After authorizing, come back and retry your command."
            }
        return {"result": f"⚠️ Backend Error: {err_str}"}


if __name__ == "__main__":
    app.run()
    