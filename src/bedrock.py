import boto3
import json

# Replace 'us-east-1' with your desired AWS region
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="eu-west-1"
)
prompt = "What is the capital of France?"

payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 2048,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
}

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
response = bedrock_runtime.invoke_model(
    modelId=model_id,
    body=json.dumps(payload)
)
response_body = json.loads(response["body"].read())
generated_text = response_body["content"][0]["text"]

print("Model response:", generated_text)

