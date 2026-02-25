import os
import json
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

# 1. THE CACHE: This keeps the "brain" in memory so it only builds ONCE
_CACHED_GRAPH = None

@app.entrypoint
def invoke(payload):
    global _CACHED_GRAPH
    
    # IMMEDIATE RESPONSE FOR PINGS (Keeps the agent from timing out on boot)
    if payload.get("type") == "warmup":
        return {"result": "Agent is warm and ready!"}

    try:
        # 2. BUILD THE BRAIN (Only happens on the VERY FIRST message)
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
                return _get_github_data()

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
            
            # Save it to memory!
            _CACHED_GRAPH = builder.compile()

        # 3. FAST EXECUTION (Happens instantly for all subsequent messages)
        from langchain_core.messages import HumanMessage
        user_message = payload.get("prompt", "Hello")
        
        res = _CACHED_GRAPH.invoke({"messages": [HumanMessage(content=user_message)]})
        return {"result": res["messages"][-1].content}

    except Exception as e:
        error_str = str(e)
        if "Auth" in error_str or "Identity" in error_str or "requires_access_token" in error_str:
            url = getattr(e, 'authorization_url', getattr(e, 'url', None))
            return {"result": f"🔒 **Permission Required:** Please [Authorize GitHub]({url})"}
        return {"result": f"⚠️ Backend Error: {error_str}"}

if __name__ == "__main__":
    app.run()
