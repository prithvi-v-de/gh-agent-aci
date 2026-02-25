import os
import json
import re
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()
_CACHED_GRAPH = None

# THE BYPASS ALARM
class AuthRequiredException(BaseException):
    def __init__(self, message):
        self.message = message

@app.entrypoint
def invoke(payload):
    global _CACHED_GRAPH
    
    if payload.get("type") == "warmup":
        return {"result": "Agent is warm and ready!"}

    try:
        if _CACHED_GRAPH is None:
            import requests
            from bedrock_agentcore.identity.auth import requires_access_token
            from langgraph.graph import StateGraph, START
            from langgraph.prebuilt import ToolNode, tools_condition
            from langchain_aws import ChatBedrockConverse
            from langchain_core.tools import tool
            from typing import TypedDict, Annotated
            from langgraph.graph.message import add_messages

            @requires_access_token(
                provider_name="github-provider",
                scopes=["repo", "read:user"],
                auth_flow='USER_FEDERATION',
                callback_url="https://gh-agent-aci-aq6nrmcvn96tycbvhnqjgf.streamlit.app" 
            )
            def _get_github_data(access_token=None):
                r = requests.get("https://api.github.com/user", headers={"Authorization": f"Bearer {access_token}"})
                if r.status_code == 200:
                    data = r.json()
                    return f"GitHub Profile: {data.get('login')} (Followers: {data.get('followers')}, Repos: {data.get('public_repos')})"
                return f"Error: {r.text}"

            @tool
            def fetch_github_profile():
                """Fetches the authenticated user's GitHub profile and stats. Requires no arguments."""
                try:
                    return _get_github_data()
                except Exception as e:
                    # AGGRESSIVE ALARM TRIGGER
                    err_str = str(e)
                    url = getattr(e, 'authorization_url', getattr(e, 'url', getattr(e, 'redirect_url', None)))
                    
                    # If it's not a direct attribute, hunt for the URL in the error string
                    if not url:
                        match = re.search(r'(https://[^\s\'">]+)', err_str)
                        if match:
                            url = match.group(1)
                    
                    if url:
                        raise AuthRequiredException(f"🔒 **Permission Required:** Please [Authorize GitHub]({url})")
                    
                    # If we still can't find a URL, break the loop and show the raw error!
                    raise AuthRequiredException(f"⚠️ **Authorization Failed:** Could not parse link. Raw Error: {err_str}")

            tools = [fetch_github_profile]
            llm = ChatBedrockConverse(model="us.anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-east-1")
            llm_with_tools = llm.bind_tools(tools)

            class State(TypedDict):
                messages: Annotated[list, add_messages]

            builder = StateGraph(State)
            builder.add_node("chatbot", lambda s: {"messages": [llm_with_tools.invoke(s["messages"])]})
            builder.add_node("tools", ToolNode(tools))
            builder.add_edge(START, "chatbot")
            builder.add_conditional_edges("chatbot", tools_condition)
            builder.add_edge("tools", "chatbot")
            
            _CACHED_GRAPH = builder.compile()

        from langchain_core.messages import HumanMessage
        user_message = payload.get("prompt", "Hello")
        
        res = _CACHED_GRAPH.invoke({"messages": [HumanMessage(content=user_message)]})
        return {"result": res["messages"][-1].content}

    except AuthRequiredException as auth_err:
        # Catch our custom alarm and return the message instantly
        return {"result": auth_err.message}
        
    except Exception as e:
        return {"result": f"⚠️ Backend Error: {str(e)}"}

if __name__ == "__main__":
    app.run()
