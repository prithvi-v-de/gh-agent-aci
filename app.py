import streamlit as st
import boto3
from botocore.config import Config
import uuid
import json

# ── Config ──
AGENT_ARN = "arn:aws:bedrock-agentcore:us-east-1:819079555973:runtime/myagent-bHKPpuHli3"

my_config = Config(
    read_timeout=600,       # 10 min — AgentCore cold starts can be slow
    connect_timeout=60,
    retries={"max_attempts": 2, "mode": "adaptive"},
)
client = boto3.client("bedrock-agentcore", region_name="us-east-1", config=my_config)

st.set_page_config(page_title="AWS Terminal", layout="centered")

# ── Terminal header ──
st.markdown("### AWS AgentCore Terminal v1.0.0")
st.markdown("Type commands below. Connection secured via IAM Identity Center.")
st.markdown("---")

# ── ──
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Switch Account"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.warmed_up = False
        st.rerun()

# ── Helper: parse the EventStream response from invoke_agent_runtime ──
def parse_agent_response(response) -> str:
    """
    invoke_agent_runtime returns a streaming EventStream.
    We need to read all chunks, concatenate, and parse the JSON envelope.
    """
    chunks = []
    event_stream = response.get("response") or response.get("ResponseStream") or response.get("body")

    if event_stream is None:
        # Fallback: try reading the response dict directly
        return response.get("result", json.dumps(response, default=str))

    try:
        for event in event_stream:
            # EventStream events can be dicts with a 'chunk' key or raw bytes
            if isinstance(event, dict):
                # e.g. {'chunk': {'bytes': b'...'}}
                chunk_data = event.get("chunk", {}).get("bytes")
                if chunk_data:
                    chunks.append(chunk_data.decode("utf-8") if isinstance(chunk_data, bytes) else str(chunk_data))
                # or the event might carry a 'bytes' key directly
                elif "bytes" in event:
                    b = event["bytes"]
                    chunks.append(b.decode("utf-8") if isinstance(b, bytes) else str(b))
                else:
                    # last resort — serialize the event
                    chunks.append(json.dumps(event, default=str))
            elif isinstance(event, bytes):
                chunks.append(event.decode("utf-8"))
            else:
                chunks.append(str(event))
    except Exception as stream_err:
        if chunks:
            pass  # use whatever we collected so far
        else:
            return f"⚠️ Stream read error: {stream_err}"

    raw = "".join(chunks)

    # Try to unwrap the JSON envelope the agent returns
    try:
        data = json.loads(raw)
        return data.get("result", raw)
    except (json.JSONDecodeError, TypeError):
        return raw if raw.strip() else "⚠️ Empty response from agent."


# ── Automatic warmup (fire once per session) ──
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
            parse_agent_response(resp)  # drain the stream
            st.session_state.warmed_up = True
            status.update(label="> ping successful. Agent is online.", state="complete")
        except Exception:
            status.update(label="> ping timeout. Agent is sleeping — first command may be slow.", state="error")

# ── Session state ──
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ──
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**guest@github-portfolio:~$** {msg['content']}")
    else:
        # Use st.markdown so OAuth links and formatting render properly
        st.markdown(f"**agent@aws:~$** {msg['content']}")

# ── User input ──
if prompt := st.chat_input("Enter command..."):
    st.markdown(f"**guest@github-portfolio:~$** {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.status("> Executing script...", expanded=True) as status:
        try:
            response = client.invoke_agent_runtime(
                agentRuntimeArn=AGENT_ARN,
                runtimeSessionId=st.session_state.session_id,
                payload=json.dumps({"prompt": prompt}).encode("utf-8"),
            )

            full_response = parse_agent_response(response)
            status.update(label="> Script executed successfully.", state="complete", expanded=False)

            # Render with markdown so links, bold, etc. work
            st.markdown(f"**agent@aws:~$** {full_response}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            err_msg = str(e)
            status.update(label="> FATAL ERROR.", state="error", expanded=True)

            # If it's a timeout, give a friendlier message
            if "timeout" in err_msg.lower() or "ReadTimeoutError" in err_msg:
                st.error(
                    "⏱️ The agent took too long to respond. This usually happens on the "
                    "first request after a cold start. Please try again."
                )
            else:
                st.error(f"Traceback: {err_msg}")
