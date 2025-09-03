import openai

# Set your API key (it's best to use an environment variable)
openai.api_key = "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"
openai.api_key = "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"
# Create a chat completion
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "predict the outcome of the formula 1 italian grand prix 7th sept 2025 all 20 drivers as csv file"}
    ]
)

# Print the assistant's response
print(response.choices[0].message.content)

