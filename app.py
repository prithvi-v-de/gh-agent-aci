import streamlit as st
import boto3
import uuid

# We will update these later
AGENT_ARN = "arn:aws:bedrock-agentcore:us-east-1:***:runtime/myagent-bHKPpuHli3"

client = boto3.client('bedrock-agent-runtime', region_name="us-east-1")

st.set_page_config(page_title="My AI Assistant", page_icon="🤖")
st.title("Talk to my AWS Agent 🤖...")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            response = client.invoke_agent(
                agentId=AGENT_ID,
                agentAliasId=AGENT_ALIAS_ID,
                sessionId=st.session_state.session_id,
                inputText=prompt
            )
            
            full_response = ""
            for event in response.get('completion'):
                if 'chunk' in event:
                    full_response += event['chunk']['bytes'].decode('utf-8')
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            message_placeholder.error(f"Error talking to AWS: {e}")
