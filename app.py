import streamlit as st
import boto3
import uuid
import json

# Replace with your actual AgentCore ARN from the AWS Console
AGENT_ARN = "arn:aws:bedrock-agentcore:us-east-1:819079555973:runtime/myagent-bHKPpuHli3"
client = boto3.client('bedrock-agentcore', region_name="us-east-1")

st.set_page_config(page_title="My AI Assistant", page_icon="🤖")
st.title("Talk to my AWS Agent 🤖")



# AUTOMATIC WARMUP
if "warmed_up" not in st.session_state:
    with st.spinner("Waking up agent..."):
        try:
            client.invoke_agent_runtime(
                agentRuntimeArn=AGENT_ARN,
                runtimeSessionId="warmup-session",
                payload=json.dumps({"type": "warmup"}).encode('utf-8')
            )
            st.session_state.warmed_up = True
            st.success("Agent is online!")
        except:
            st.warning("Agent is still sleeping. First message might take a moment.")


# Initialize Session state for Chat
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display persistent chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input processing
if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # The spinner keeps the user engaged during the 15-20s LangGraph boot
        with st.status("🤖 Agent is waking up...", expanded=True) as status:
            try:
                response = client.invoke_agent_runtime(
                    agentRuntimeArn=AGENT_ARN,
                    runtimeSessionId=st.session_state.session_id,
                    payload=json.dumps({"prompt": prompt}).encode('utf-8')
                )
                
                # Assemble the response from the streaming chunks
                content_list = []
                for chunk in response.get('response', []):
                    content_list.append(chunk.decode('utf-8'))
                
                raw_json = "".join(content_list)
                result_dict = json.loads(raw_json)
                full_response = result_dict.get("result", "No response found.")
                
                status.update(label="Response received!", state="complete", expanded=False)
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                status.update(label="Connection Error", state="error", expanded=True)
                st.error(f"Error talking to AWS: {e}")
