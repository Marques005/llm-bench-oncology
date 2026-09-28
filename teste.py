from huggingface_hub import InferenceClient
client = InferenceClient()
r = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is a patient-derived xenograft? One sentence."}],
    max_tokens=60,
)
print(r.choices[0].message.content)