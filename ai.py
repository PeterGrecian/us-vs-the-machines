from transformers import pipeline

# Create a text generation pipeline with a small, easy-to-run model
generator = pipeline("text-generation", model="gpt-4o-mini")

# Generate text
result = generator("write a haiku about ai", max_length=50, num_return_sequences=1)

# Print the generated text
print(result[0]['generated_text'])

