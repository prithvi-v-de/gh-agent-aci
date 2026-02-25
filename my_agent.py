import os
import json
import requests
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.identity.auth import requires_access_token

# We keep global variables for the graph so we only build it ONCE per 'warm' session
_AGENT_GRAPH = None

def get_graph():
    """Lazily initialize the LangGraph brain to beat the 30s timeout."""
    global _AGENT_GRAPH
    if _AGENT_GRAPH is not None:
        return _AGENT_GRAPH

    # Heavy imports stay inside here!
    from langgraph.graph import StateGraph, START
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_aws import ChatBedrockConverse
    from langchain_core.tools import tool
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages

    @tool
    def get_github_user_profile():
        """Fetches the authenticated user's GitHub profile."""
        return _internal_github_profile()

    tools = [get_github_user_profile]
    llm = ChatBedrockConverse(
        model="us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    builder = StateGraph(State)
    builder.add_node("chatbot", lambda s: {"messages": [llm.bind_tools(tools).invoke(s["messages"])]})
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    
    _AGENT_GRAPH = builder.compile()
    return _AGENT_GRAPH

@requires_access_token(
    provider_name="github-provider",
    scopes=["repo", "read:user"],
    auth_flow='USER_FEDERATION',
    return_url="YOUR_STREAMLIT_URL_HERE"
)
def _internal_github_profile(access_token=None):
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.get("https://api.github.com/user", headers=headers)
    return r.json() if r.status_code == 200 else f"Error: {r.text}"

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    try:
        from langchain_core.messages import HumanMessage
        agent_graph = get_graph()
        user_message = payload.get("prompt", "Hello")
        result = agent_graph.invoke({"messages": [HumanMessage(content=user_message)]})
        return {"result": result["messages"][-1].content}
    except Exception as e:
        # Same error handling we built before to catch the OAuth link
        error_name = type(e).__name__
        if "Auth" in error_name or "Identity" in error_name:
            auth_url = getattr(e, 'authorization_url', getattr(e, 'url', None))
            return {"result": f"🔒 **Permission Required:** Please [Authorize GitHub]({auth_url})"}
        return {"result": f"⚠️ **Backend Error:** {str(e)}"}

if __name__ == "__main__":
    app.run()
