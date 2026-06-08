import boto3
import os
import sys

client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")


def stream_chat(user_message: str, system_prompt: str = "You are a helpful assistant.") -> str:
    response = client.converse_stream(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{
            "role": "user",
            "content": [{"text": user_message}]
        }],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.7}
    )

    full_response = ""
    print("Bot: ", end="", flush=True)

    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                chunk = delta["text"]
                print(chunk, end="", flush=True)
                full_response += chunk

        if "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "")
            if stop_reason == "max_tokens":
                print("\n[Warning: response truncated]", end="")

        if "metadata" in event:
            usage = event["metadata"].get("usage", {})
            print(f"\n[tokens] input={usage.get('inputTokens', 0)} output={usage.get('outputTokens', 0)}")

    print()
    return full_response


def stream_with_history(messages: list, system_prompt: str = "You are a helpful assistant.") -> str:
    response = client.converse_stream(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=messages,
        inferenceConfig={"maxTokens": 2048, "temperature": 0.7}
    )

    full_response = ""
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                chunk = delta["text"]
                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_response += chunk

    print()
    return full_response


if __name__ == "__main__":
    print(f"Streaming chatbot ready. Model: {MODEL_ID}")
    print("Type 'exit' to quit.\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        history.append({
            "role": "user",
            "content": [{"text": user_input}]
        })

        reply = stream_with_history(history)
        history.append({
            "role": "assistant",
            "content": [{"text": reply}]
        })
