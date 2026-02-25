import os
import requests
import traceback
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.identity.auth import requires_access_token
from langgraph.prebuilt import ToolNode, tools_condition

@requires_access_token(
    provider_name="github-provider",
    scopes=["repo", "read:user"],
    auth_flow='USER_FEDERATION',
    return_url="https://gh-agent-aci-aq6nrmcvn96tycbvhnqjgf.streamlit.app/"
)
def _internal_github_profile(access_token=None):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get("https://api.github.com/user", headers=headers)
    if response.status_code == 200:
        data = response.json()
        return f"User: {data.get('login')}, Followers: {data.get('followers')}"
    return f"Failed to fetch profile: {response.text}"

@tool
def get_github_user_profile():
    """Fetches the authenticated user's GitHub profile. Requires no arguments."""
    return _internal_github_profile()

tools = [get_github_user_profile]

llm = ChatBedrockConverse(
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
    region_name=os.getenv("AWS_REGION", "us-east-1")
)
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

agent_graph = graph_builder.compile()

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    try:
        user_message = payload.get("prompt", "Hello")
        result = agent_graph.invoke({"messages": [HumanMessage(content=user_message)]})
        return {"result": result["messages"][-1].content}
        
    except Exception as e:
        error_name = type(e).__name__
        traceback.print_exc() # Still logs to CloudWatch just in case
        
        if "Auth" in error_name or "Authorization" in error_name or "Identity" in error_name:
            auth_url = getattr(e, 'authorization_url', getattr(e, 'url', None))
            if auth_url:
                return {"result": f"🔒 **Permission Required:** Please [Click here to Authorize]({auth_url})"}
            return {"result": f"🔒 Auth required, but couldn't extract URL. Error details: {str(e)}"}
        
        return {"result": f"⚠️ **Backend Crash:** {error_name} - {str(e)}"}

if __name__ == "__main__":
    app.run()
