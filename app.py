import os
import json
import uuid
import boto3
from botocore.config import Config
from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "super_secret_dev_key_123")

AGENT_ARN = "arn:aws:bedrock-agentcore:us-east-1:819079555973:runtime/myagent-bHKPpuHli3"
AUTH_PREFIX = "__AUTH_REQUIRED__"

my_config = Config(
    read_timeout=600,
    connect_timeout=60,
    retries={"max_attempts": 2, "mode": "adaptive"},
)
client = boto3.client("bedrock-agentcore", region_name="us-east-1", config=my_config)


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
    # Generate session ID if not exists
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    # Check if this is an OAuth callback redirect
    oauth_callback = "session_id" in request.args
    return render_template("index.html", oauth_callback=oauth_callback)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    
    # 1. Prefer the unbreakable session_id from the frontend localStorage
    client_session_id = data.get("session_id")
    if not client_session_id:
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        client_session_id = session["session_id"]

    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_ARN,
            runtimeSessionId=client_session_id, # Use the synced ID
            payload=json.dumps({"prompt": prompt}).encode("utf-8"),
        )
        result = parse_agent_response(response)

        # Check if auth is required
        if result.startswith(AUTH_PREFIX):
            auth_url = result[len(AUTH_PREFIX):]
            return jsonify({"type": "auth", "url": auth_url})

        return jsonify({"type": "response", "text": result})

    except Exception as e:
        return jsonify({"type": "error", "text": str(e)}), 500


@app.route("/api/warmup", methods=["POST"])
def warmup():
    try:
        resp = client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_ARN,
            # Generate a full 36-character string to satisfy AWS constraints
            runtimeSessionId=str(uuid.uuid4()), 
            payload=json.dumps({"type": "warmup"}).encode("utf-8"),
        )
        parse_agent_response(resp)
        return jsonify({"status": "online"})
    except Exception:
        return jsonify({"status": "sleeping"}), 503


@app.route("/api/reset", methods=["POST"])
def reset():
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "ok", "session_id": session["session_id"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
