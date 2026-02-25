import os
import json
import requests
import traceback
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.identity.auth import requires_access_token

# Global variable to cache the compiled graph so it only builds once
_CACHED_GRAPH = None

def get_agent_graph():
    """Builds the LangGraph brain only when called to bypass cold-start timeouts."""
    global _CACHED_GRAPH
    if _CACHED_GRAPH:
        return _CACHED_GRAPH

    # Heavy imports are moved inside the function to keep the initial boot under 30s
    from langgraph.graph import StateGraph, START
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_aws import ChatBedrockConverse
    from langchain_core.tools import tool
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage

    @tool
    def get_github_user_profile():
        """Fetches the authenticated user's GitHub profile. Requires no arguments."""
        return _internal_github_profile()

    tools = [get_github_user_profile]
    
    # Initialize Claude 3.5 Sonnet
    llm = ChatBedrockConverse(
        model="us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    llm_with_tools = llm.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def chatbot_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Define the graph structure
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    
    _CACHED_GRAPH = builder.compile()
    return _CACHED_GRAPH

@requires_access_token(
    provider_name="github-provider",
    scopes=["repo", "read:user"],
    auth_flow='USER_FEDERATION',
    return_url="https://gh-agent-aci-aq6nrmcvn96tycbvhnqjgf.streamlit.app/" 
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
    try:
        from langchain_core.messages import HumanMessage
        user_message = payload.get("prompt", "Hello")
        
        # This triggers the heavy LangGraph build ONLY after the agent reports 'Ready' to AWS
        graph = get_agent_graph() 
        result = graph.invoke({"messages": [HumanMessage(content=user_message)]})
        
        return {"result": result["messages"][-1].content}
        
    except Exception as e:
        error_name = type(e).__name__
        # Specifically catch AgentCore's identity exception to extract the Auth URL
        if "Auth" in error_name or "Identity" in error_name or "Authorization" in error_name:
            auth_url = getattr(e, 'authorization_url', getattr(e, 'url', None))
            if auth_url:
                return {"result": f"🔒 **Permission Required:** Please [Authorize GitHub]({auth_url})"}
        
        # Log the error to CloudWatch and show it in Streamlit
        traceback.print_exc()
        return {"result": f"⚠️ **Backend Error:** {error_name} - {str(e)}"}

if __name__ == "__main__":
    app.run()
