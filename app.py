import os
import json
import boto3
from botocore.config import Config
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
# We don't even need a secret key anymore because we aren't using Flask sessions!

AGENT_ARN = "arn:aws:bedrock-agentcore:us-east-2:819079555973:runtime/myagent-WwnsnQFwSt"
AUTH_PREFIX = "__AUTH_REQUIRED__"

my_config = Config(
    read_timeout=600,
    connect_timeout=60,
    retries={"max_attempts": 2, "mode": "adaptive"},
)
client = boto3.client("bedrock-agentcore", region_name="us-east-2", config=my_config)

def parse_agent_response(response) -> str:
    chunks = []
    event_stream = (
        response.get("response")
        or response.get("ResponseStream")
        or response.get("body")
    )
    if event_stream is None:
        return response.get("result", json.dumps(response, default=str))

    try:
        for event in event_stream:
            if isinstance(event, dict):
                chunk_data = event.get("chunk", {}).get("bytes")
                if chunk_data:
                    chunks.append(
                        chunk_data.decode("utf-8")
                        if isinstance(chunk_data, bytes)
                        else str(chunk_data)
                    )
                elif "bytes" in event:
                    b = event["bytes"]
                    chunks.append(
                        b.decode("utf-8") if isinstance(b, bytes) else str(b)
                    )
                else:
                    chunks.append(json.dumps(event, default=str))
            elif isinstance(event, bytes):
                chunks.append(event.decode("utf-8"))
            else:
                chunks.append(str(event))
    except Exception:
        if not chunks:
            return "Stream read error"

    raw = "".join(chunks)
    try:
        data = json.loads(raw)
        return data.get("result", raw)
    except (json.JSONDecodeError, TypeError):
        return raw if raw.strip() else "Empty response."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    session_uri = data.get("session_uri")  # NEW: from frontend
    
    client_session_id = data.get("session_id")
    if not client_session_id or len(client_session_id) < 33:
        return jsonify({"type": "error", "text": "Invalid session ID."}), 400

    try:
        # Include session_uri in payload to agent
        agent_payload = {"prompt": prompt}
        if session_uri:
            agent_payload["session_uri"] = session_uri

        response = client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_ARN,
            runtimeSessionId=client_session_id,
            payload=json.dumps(agent_payload).encode("utf-8"),
        )
        raw = parse_agent_response(response)

        # Try to parse as JSON to extract session_uri
        try:
            parsed = json.loads(raw)
            result = parsed.get("result", raw)
            resp_session_uri = parsed.get("session_uri")
        except (json.JSONDecodeError, TypeError):
            result = raw
            resp_session_uri = None

        if result.startswith(AUTH_PREFIX):
            auth_url = result[len(AUTH_PREFIX):]
            resp = {"type": "auth", "url": auth_url}
            if resp_session_uri:
                resp["session_uri"] = resp_session_uri  # NEW: pass to frontend
            return jsonify(resp)

        return jsonify({"type": "response", "text": result})

    except Exception as e:
        return jsonify({"type": "error", "text": str(e)}), 500

@app.route("/api/warmup", methods=["POST"])
def warmup():
    data = request.json or {}
    client_session_id = data.get("session_id")
    if not client_session_id or len(client_session_id) < 33:
        return jsonify({"status": "error"}), 400

    try:
        resp = client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_ARN,
            runtimeSessionId=client_session_id,
            payload=json.dumps({"type": "warmup"}).encode("utf-8"),
        )
        parse_agent_response(resp)
        return jsonify({"status": "online"})
    except Exception:
        return jsonify({"status": "sleeping"}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
