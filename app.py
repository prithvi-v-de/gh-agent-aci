import streamlit as st
import boto3
from botocore.config import Config
import uuid
import json
import hashlib

# ── Config ──
AGENT_ARN = "arn:aws:bedrock-agentcore:us-east-1:819079555973:runtime/myagent-bHKPpuHli3"

my_config = Config(
    read_timeout=600,
    connect_timeout=60,
    retries={"max_attempts": 2, "mode": "adaptive"},
)
client = boto3.client("bedrock-agentcore", region_name="us-east-1", config=my_config)

st.set_page_config(page_title="AWS Terminal", layout="centered")

# ── Detect OAuth redirect ──
# After GitHub authorization, AgentCore redirects back here with a session_id param
query_params = st.query_params
oauth_callback = "session_id" in query_params

# ── Terminal header ──
st.markdown("### AWS AgentCore Terminal v1.0.0")
st.markdown("Type commands below. Connection secured via IAM Identity Center.")
st.markdown("---")

# ── Switch account ──
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Switch Account"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.warmed_up = False
        st.query_params.clear()
        st.rerun()


def parse_agent_response(response) -> str:
    chunks = []
    event_stream = response.get("response") or response.get("ResponseStream") or response.get("body")

    if event_stream is None:
        return response.get("result", json.dumps(response, default=str))

    try:
        for event in event_stream:
            if isinstance(event, dict):
                chunk_data = event.get("chunk", {}).get("bytes")
                if chunk_data:
                    chunks.append(chunk_data.decode("utf-8") if isinstance(chunk_data, bytes) else str(chunk_data))
                elif "bytes" in event:
                    b = event["bytes"]
                    chunks.append(b.decode("utf-8") if isinstance(b, bytes) else str(b))
                else:
                    chunks.append(json.dumps(event, default=str))
            elif isinstance(event, bytes):
                chunks.append(event.decode("utf-8"))
            else:
                chunks.append(str(event))
    except Exception:
        if not chunks:
            return "⚠️ Stream read error"

    raw = "".join(chunks)
    try:
        data = json.loads(raw)
        return data.get("result", raw)
    except (json.JSONDecodeError, TypeError):
        return raw if raw.strip() else "⚠️ Empty response from agent."


def invoke_agent(prompt, session_id):
    """Call the agent and return the response string."""
    response = client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_ARN,
        runtimeSessionId=session_id,
        payload=json.dumps({"prompt": prompt}).encode("utf-8"),
    )
    return parse_agent_response(response)


# ── Warmup ──
if "warmed_up" not in st.session_state:
    st.session_state.warmed_up = False

if not st.session_state.warmed_up:
    with st.status("> ping -c 1 bedrock-agentcore...", expanded=False) as status:
        try:
            resp = client.invoke_agent_runtime(
                agentRuntimeArn=AGENT_ARN,
                runtimeSessionId="warmup-" + str(uuid.uuid4())[:8],
                payload=json.dumps({"type": "warmup"}).encode("utf-8"),
            )
            parse_agent_response(resp)
            st.session_state.warmed_up = True
            status.update(label="> ping successful. Agent is online.", state="complete")
        except Exception:
            status.update(label="> ping timeout. Agent is sleeping — first command may be slow.", state="error")

# ── Session state ──
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Handle OAuth callback redirect ──
if oauth_callback and not st.session_state.get("oauth_handled"):
    st.session_state.oauth_handled = True
    # Clear the URL params so refresh doesn't re-trigger
    st.query_params.clear()

    st.success("✅ GitHub authorization successful! Fetching your profile...")

    # Auto-retry the github fetch
    try:
        full_response = invoke_agent("fetch my github profile", st.session_state.session_id)
        st.session_state.messages.append({"role": "user", "content": "fetch my github profile"})
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
    except Exception as e:
        st.warning(f"Auto-fetch failed: {e}. Type 'fetch my github profile' to try again.")

# ── Render chat history ──
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**guest@github-portfolio:~$** {msg['content']}")
    else:
        st.markdown(f"**agent@aws:~$** {msg['content']}")

# ── User input ──
if prompt := st.chat_input("Enter command..."):
    st.markdown(f"**guest@github-portfolio:~$** {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Reset oauth_handled so future auth flows work
    st.session_state.oauth_handled = False

    with st.status("> Executing script...", expanded=True) as status:
        try:
            full_response = invoke_agent(prompt, st.session_state.session_id)
            status.update(label="> Script executed successfully.", state="complete", expanded=False)

            st.markdown(f"**agent@aws:~$** {full_response}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            err_msg = str(e)
            status.update(label="> FATAL ERROR.", state="error", expanded=True)
            if "timeout" in err_msg.lower() or "ReadTimeoutError" in err_msg:
                st.error("⏱️ The agent took too long to respond. Please try again.")
            else:
                st.error(f"Traceback: {err_msg}")