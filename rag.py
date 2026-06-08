import boto3
import os

client = boto3.client("bedrock-agent-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
MODEL_ARN = f"arn:aws:bedrock:{os.getenv('AWS_REGION', 'us-east-1')}::foundation-model/{os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')}"


def query_knowledge_base(question: str, num_results: int = 5) -> dict:
    if not KNOWLEDGE_BASE_ID:
        raise ValueError("KNOWLEDGE_BASE_ID environment variable not set")

    response = client.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ARN,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": num_results
                    }
                }
            }
        }
    )

    answer = response["output"]["text"]
    citations = response.get("citations", [])

    sources = []
    for citation in citations:
        for ref in citation.get("retrievedReferences", []):
            sources.append({
                "text": ref["content"]["text"][:200],
                "location": ref.get("location", {})
            })

    return {"answer": answer, "sources": sources}


def retrieve_only(question: str, num_results: int = 5) -> list:
    runtime_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )

    response = runtime_client.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": num_results}
        }
    )

    return [
        {
            "text": r["content"]["text"],
            "score": r.get("score", 0),
            "location": r.get("location", {})
        }
        for r in response.get("retrievalResults", [])
    ]


if __name__ == "__main__":
    question = input("Ask a question: ").strip()
    result = query_knowledge_base(question)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for i, s in enumerate(result["sources"], 1):
        print(f"  {i}. {s['text']}...")
