import boto3
import os

client = boto3.client("bedrock-agent-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

AGENT_ID = os.getenv("AGENT_ID")
AGENT_ALIAS_ID = os.getenv("AGENT_ALIAS_ID", "TSTALIASID")


def invoke_agent(prompt: str, session_id: str = "session-001") -> str:
    if not AGENT_ID:
        raise ValueError("AGENT_ID environment variable not set")

    response = client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=prompt
    )

    full_response = ""
    for event in response["completion"]:
        if "chunk" in event:
            chunk = event["chunk"]["bytes"].decode("utf-8")
            full_response += chunk
            print(chunk, end="", flush=True)

    print()
    return full_response


def invoke_agent_with_trace(prompt: str, session_id: str = "session-001") -> dict:
    response = client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=prompt,
        enableTrace=True
    )

    full_response = ""
    traces = []

    for event in response["completion"]:
        if "chunk" in event:
            full_response += event["chunk"]["bytes"].decode("utf-8")
        if "trace" in event:
            traces.append(event["trace"])

    return {"response": full_response, "traces": traces}


if __name__ == "__main__":
    print(f"Agent ready. ID: {AGENT_ID}")
    print("Type 'exit' to quit.\n")

    session = f"session-{os.getpid()}"
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        print("Agent: ", end="")
        invoke_agent(user_input, session_id=session)
        print()
