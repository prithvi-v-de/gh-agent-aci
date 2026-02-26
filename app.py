import streamlit as st
import boto3
from botocore.config import Config
import uuid
import json

AGENT_ARN = "arn:aws:bedrock-agentcore:us-east-1:819079555973:runtime/myagent-bHKPpuHli3"

my_config = Config(
    read_timeout=300,
    connect_timeout=300,
    retries={'max_attempts': 0}
)
client = boto3.client('bedrock-agentcore', region_name="us-east-1", config=my_config)

st.set_page_config(page_title="AWS Terminal", layout="centered")

st.markdown("""
<style>
    .stApp {
        background-color: #0c0c0c;
    }
    html, body, [class*="css"], p, div, span {
        font-family: 'Courier New', Courier, monospace !important;
        color: #00FF00 !important;
    }
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Style the input box to look like a command line */
    [data-testid="stChatInput"] {
        background-color: #0c0c0c !important;
        border-top: 1px solid #00FF00 !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #00FF00 !important;
    }
    
    /* Agent output color (Cyan for contrast) */
    .agent-text {
        color: #00FFFF !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("### AWS Bedrock AgentCore Terminal v1.0.0")
st.markdown("Type commands below. Connection secured via IAM Identity Center.")
st.markdown("---")

if "warmed_up" not in st.session_state:
    with st.status("> ping -c 1 bedrock-agentcore...", expanded=False) as status:
        try:
            client.invoke_agent_runtime(
                agentRuntimeArn=AGENT_ARN,
                runtimeSessionId="warmup-session-" + str(uuid.uuid4())[:8],
                payload=json.dumps({"type": "warmup"}).encode('utf-8')
            )
            st.session_state.warmed_up = True
            status.update(label="> ping successful. Agent is online.", state="complete")
        except Exception as e:
            status.update(label="> ping timeout. Agent is sleeping.", state="error")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# We use raw markdown instead of st.chat_message to remove the bubbles/avatars
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**guest@github-portfolio:~$** {message['content']}")
    else:
        st.markdown(f"<span class='agent-text'>**agent@aws:~$** {message['content']}</span>", unsafe_allow_html=True)

if prompt := st.chat_input("Enter command..."):
    # Display user command immediately
    st.markdown(f"**guest@github-portfolio:~$** {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.status("> Executing script...", expanded=True) as status:
        try:
            response = client.invoke_agent_runtime(
                agentRuntimeArn=AGENT_ARN,
                runtimeSessionId=st.session_state.session_id,
                payload=json.dumps({"prompt": prompt}).encode('utf-8')
            )
            
            content_list = []
            for chunk in response.get('response', []):
                content_list.append(chunk.decode('utf-8'))
            
            raw_json = "".join(content_list)
            result_dict = json.loads(raw_json)
            full_response = result_dict.get("result", "No response found.")
            
            status.update(label="> Script executed successfully.", state="complete", expanded=False)
            
            st.markdown(f"<span class='agent-text'>**agent@aws:~$** {full_response}</span>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            status.update(label="> FATAL ERROR.", state="error", expanded=True)
            st.error(f"Traceback: {e}")
