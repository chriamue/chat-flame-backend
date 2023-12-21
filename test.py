from huggingface_hub import InferenceClient

client = InferenceClient(model="http://127.0.0.1:8080")
for token in client.text_generation("How do you make cheese?", max_new_tokens=42, details=True, stream=True):
    print(token)
