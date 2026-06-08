import boto3
import os

client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

chat_history = []


def chat(user_message: str) -> str:
    chat_history.append({
        "role": "user",
        "content": [{"text": user_message}]
    })

    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=chat_history,
        inferenceConfig={"maxTokens": 1024, "temperature": 0.7}
    )

    reply = response["output"]["message"]["content"][0]["text"]
    chat_history.append({
        "role": "assistant",
        "content": [{"text": reply}]
    })

    # Token usage logging
    usage = response.get("usage", {})
    print(f"[tokens] input={usage.get('inputTokens', 0)} output={usage.get('outputTokens', 0)}")

    return reply


def reset():
    global chat_history
    chat_history = []
    print("Chat history cleared.")


if __name__ == "__main__":
    print(f"Chatbot ready. Model: {MODEL_ID}")
    print("Type 'exit' to quit, 'reset' to clear history.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "reset":
            reset()
            continue
        reply = chat(user_input)
        print(f"Bot: {reply}\n")
