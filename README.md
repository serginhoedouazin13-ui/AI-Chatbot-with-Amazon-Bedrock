# AI-Chatbot-with-Amazon-Bedrock
A conversational AI chatbot built on Amazon Bedrock, giving you access to powerful foundation models (Claude, Llama, Titan, Mistral, and more) through a unified API — with optional support for Retrieval-Augmented Generation (RAG) and multi-step agentic workflows.

## Features

-Multi-turn conversational chat with persistent message history

-Choice of foundation models via the Bedrock Converse API

-Optional Knowledge Base integration for RAG (grounded answers from your own documents)

-Optional Bedrock Agents for tool use and multi-step task execution

-Built-in content filtering with Bedrock Guardrails

-Streaming response support

-CloudWatch logging and token usage tracking

---

## Prerequisites
 
- Python 3.9+ (or Node.js 18+ for JS SDK)
- An AWS account with Bedrock access
- AWS CLI configured (`aws configure`)
- IAM role or user with the following permissions:
  - `bedrock:InvokeModel`
  - `bedrock:Converse`
  - `bedrock:Retrieve` *(if using Knowledge Bases)*
  - `bedrock:RetrieveAndGenerate` *(if using Knowledge Bases)*
 
---
 
## Setup
 
### 1. Enable model access
 
In the AWS Console, navigate to **Amazon Bedrock → Model access** and enable the models you want to use. This step is required before any API calls will succeed.
 
### 2. Install dependencies
 
```bash
pip install boto3
```
 
### 3. Configure AWS credentials
 
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and preferred region (e.g. us-east-1)
```
 
---
 
## Quickstart
 
```python
import boto3
 
client = boto3.client("bedrock-runtime", region_name="us-east-1")
 
chat_history = []
 
def chat(user_message):
    chat_history.append({"role": "user", "content": [{"text": user_message}]})
 
    response = client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        system=[{"text": "You are a helpful assistant."}],
        messages=chat_history,
        inferenceConfig={"maxTokens": 1024, "temperature": 0.7},
    )
 
    reply = response["output"]["message"]["content"][0]["text"]
    chat_history.append({"role": "assistant", "content": [{"text": reply}]})
    return reply
 
# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    print("Bot:", chat(user_input))
```
 
---
 
## Supported Models
 
| Model | Model ID |
|---|---|
| Claude 3 Sonnet | `anthropic.claude-3-sonnet-20240229-v1:0` |
| Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` |
| Claude 3 Opus | `anthropic.claude-3-opus-20240229-v1:0` |
| Llama 3 8B | `meta.llama3-8b-instruct-v1:0` |
| Titan Text G1 | `amazon.titan-text-express-v1` |
| Mistral 7B | `mistral.mistral-7b-instruct-v0:2` |
 
> Model availability varies by AWS region. Check the [Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for a full list.
 
---
 
## Knowledge Base (RAG)
 
To ground the chatbot in your own documents:
 
1. Upload your files (PDF, HTML, TXT) to an S3 bucket.
2. Create a Knowledge Base in the AWS Console under **Bedrock → Knowledge bases** and sync it to your S3 bucket. Bedrock will chunk, embed, and store the content in a vector store (OpenSearch Serverless by default).
3. Use `RetrieveAndGenerate` instead of `Converse`:
 
```python
response = client.retrieve_and_generate(
    input={"text": "What is our refund policy?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "YOUR_KB_ID",
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        },
    },
)
print(response["output"]["text"])
```
 
---
 
## Agents (Tool Use)
 
Bedrock Agents allow the model to call external tools (APIs, databases, Lambda functions) to complete multi-step tasks.
 
1. Create an Agent in the AWS Console and attach an IAM role.
2. Define action groups backed by Lambda functions.
3. Invoke the agent from your application:
 
```python
agent_client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
 
response = agent_client.invoke_agent(
    agentId="YOUR_AGENT_ID",
    agentAliasId="YOUR_ALIAS_ID",
    sessionId="session-001",
    inputText="Book a meeting for tomorrow at 2pm",
)
 
for event in response["completion"]:
    if "chunk" in event:
        print(event["chunk"]["bytes"].decode(), end="", flush=True)
```
 
---
 
## Streaming Responses
 
Use `converse_stream` for real-time token streaming:
 
```python
response = client.converse_stream(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{"role": "user", "content": [{"text": "Tell me a story."}]}],
    inferenceConfig={"maxTokens": 1024},
)
 
for event in response["stream"]:
    if "contentBlockDelta" in event:
        print(event["contentBlockDelta"]["delta"]["text"], end="", flush=True)
```
 
---
 
## Guardrails
 
Attach a Guardrail to filter harmful content or enforce topic restrictions:
 
```python
response = client.converse(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=chat_history,
    guardrailConfig={
        "guardrailIdentifier": "YOUR_GUARDRAIL_ID",
        "guardrailVersion": "DRAFT",
        "trace": "enabled",
    },
)
```
 
Create and configure guardrails in the AWS Console under **Bedrock → Guardrails**.
 
---
 
## Project Structure
 
```
.
├── chatbot.py          # Core chat loop using the Converse API
├── rag.py              # Knowledge Base / RAG integration
├── agents.py           # Bedrock Agents integration
├── streaming.py        # Streaming response example
└── README.md
```
 
---
 
## Environment Variables
 
| Variable | Description |
|---|---|
| `AWS_REGION` | AWS region (e.g. `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | AWS access key (if not using IAM role) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key (if not using IAM role) |
| `BEDROCK_MODEL_ID` | Default model ID to use |
| `KNOWLEDGE_BASE_ID` | Bedrock Knowledge Base ID (optional) |
| `AGENT_ID` | Bedrock Agent ID (optional) |
 
---
 
## Monitoring
 
Enable CloudWatch logging in the Bedrock console to track:
 
- Request and response logs
- Token usage (input / output / total)
- Latency per invocation
- Guardrail trace results
 
---
 
## Resources
 
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Converse API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- [Supported Foundation Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)
- [Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
 
---
 
## License
 
MIT
