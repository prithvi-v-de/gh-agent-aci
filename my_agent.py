import os
import json
from bedrock_agentcore import BedrockAgentCoreApp

# We don't even import the security decorator at the top level
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    # 1. IMMEDIATE RESPONSE FOR PINGS
    # This wakes up the server without loading any AI libraries
    if payload.get("type") == "warmup":
        return {"result": "Agent is warm and ready!"}

    try:
        # 2. DELAYED HEAVY IMPORTS
        # These only run AFTER the 'Ready' signal is sent to AWS
        import requests
        from bedrock_agentcore.identity.auth import requires_access_token
        from langgraph.graph import StateGraph, START
        from langchain_aws import ChatBedrockConverse
        from langchain_core.messages import HumanMessage
        
        user_message = payload.get("prompt", "Hello")

        # Define the internal tool inside the invoke to keep it encapsulated
        @requires_access_token(
            provider_name="github-provider",
            scopes=["repo", "read:user"],
            auth_flow='USER_FEDERATION',
            callback_url="https://gh-agent-aci-aq6nrmcvn96tycbvhnqjgf.streamlit.app" 
        )
        def _get_github_data(access_token=None):
            r = requests.get("https://api.github.com/user", headers={"Authorization": f"Bearer {access_token}"})
            return r.json() if r.status_code == 200 else f"Error: {r.text}"

        # Build a temporary graph for this session
        # (Once warm, this happens in ~2 seconds)
        llm = ChatBedrockConverse(model="us.anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-east-1")
        builder = StateGraph(dict)
        builder.add_node("chatbot", lambda s: {"reply": llm.invoke(s["msg"])})
        builder.add_edge(START, "chatbot")
        graph = builder.compile()

        res = graph.invoke({"msg": [HumanMessage(content=user_message)]})
        return {"result": res["reply"].content}

    except Exception as e:
        # Catch the Auth URL even if it's buried deep
        error_str = str(e)
        if "Auth" in error_str or "Identity" in error_str:
            url = getattr(e, 'authorization_url', None)
            return {"result": f"🔒 Please [Authorize GitHub]({url})"}
        return {"result": f"⚠️ Backend Busy: {error_str}"}

if __name__ == "__main__":
    app.run()
