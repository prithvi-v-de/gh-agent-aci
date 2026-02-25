import os
import json
import requests
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.identity.auth import requires_access_token

# We use a global variable to keep the brain in memory once it's loaded
_CACHED_GRAPH = None

@requires_access_token(
    provider_name="github-provider",
    scopes=["repo", "read:user"],
    auth_flow='USER_FEDERATION',
    return_url="https://your-app.streamlit.app" # <--- CHECK THIS URL!
)
def _internal_github_profile(access_token=None):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get("https://api.github.com/user", headers=headers)
    if response.status_code == 200:
        data = response.json()
        return f"GitHub Profile: {data.get('login')} (Followers: {data.get('followers')})"
    return f"Failed to fetch profile: {response.text}"

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    global _CACHED_GRAPH
    try:
        # Move HEAVY imports here so they don't block the 30s boot timer
        from langgraph.graph import StateGraph, START
        from langgraph.prebuilt import ToolNode, tools_condition
        from langchain_aws import ChatBedrockConverse
        from langchain_core.tools import tool
        from langchain_core.messages import HumanMessage
        from typing import TypedDict, Annotated
        from langgraph.graph.message import add_messages

        user_message = payload.get("prompt", "Hello")

        # Only build the graph if it doesn't exist in the current 'warm' session
        if _CACHED_GRAPH is None:
            @tool
            def get_github_user_profile():
                """Fetches the authenticated user's GitHub profile."""
                return _internal_github_profile()

            tools = [get_github_user_profile]
            llm = ChatBedrockConverse(
                model="us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
                region_name="us-east-1"
            )
            
            class State(TypedDict):
                messages: Annotated[list, add_messages]

            builder = StateGraph(State)
            builder.add_node("chatbot", lambda s: {"messages": [llm.bind_tools(tools).invoke(s["messages"])]})
            builder.add_node("tools", ToolNode(tools))
            builder.add_edge(START, "chatbot")
            builder.add_conditional_edges("chatbot", tools_condition)
            builder.add_edge("tools", "chatbot")
            _CACHED_GRAPH = builder.compile()

        result = _CACHED_GRAPH.invoke({"messages": [HumanMessage(content=user_message)]})
        return {"result": result["messages"][-1].content}
        
    except Exception as e:
        error_name = type(e).__name__
        if "Auth" in error_name or "Identity" in error_name:
            auth_url = getattr(e, 'authorization_url', getattr(e, 'url', None))
            return {"result": f"🔒 **Permission Required:** Please [Authorize GitHub]({auth_url})"}
        return {"result": f"⚠️ **Backend Error:** {str(e)}"}

if __name__ == "__main__":
    app.run()
