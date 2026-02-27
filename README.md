# gh-agent-aci
deploys a gh agent unto the target profile to retrieve details i.e. agent authenticates with GitHub via OAuth (user federation), fetches the user's profile, repos, and language stats, and uses Claude to explain the code and activity on the account.


FE UI  →  Bedrock AgentCore Runtime  →  LangGraph Agent
                                                   ├── Claude 3.5 Sonnet (LLM)
                                                   └── fetch_github_profile (tool)
                                                          └── GitHub API (OAuth via AgentCore Identity)



Prereqs

*AWS Account with Bedrock AgentCore enabled in us-east-1
*AgentCore Identity provider named github-provider configured with GitHub OAuth app credentials
*GitHub OAuth App with the callback URL set to your Streamlit app URL

GitHub Secrets:  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY


Deployment - 
*Install deps
*Configure AWS creds [SSO/Perm]
*Deploys the agent via agentcore launch




SDK: get_resource_oauth2_token → gets authorizationUrl + sessionUri  
SDK: on_auth_url(url) → stores URL, returns normally ✅
SDK: sessionUri captured in req ✅
SDK: _QuickTokenPoller.poll_for_token() → waits 1s → raises AuthRequiredException ✅
User: clicks auth link → authorizes → redirected back
Next call: get_resource_oauth2_token → returns accessToken directly ✅ (line 214)