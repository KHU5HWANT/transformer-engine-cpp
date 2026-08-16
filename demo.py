import requests
import sys

prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time, "
print(f"{prompt}", end="", flush=True)

# Generate up to 50 characters for a story!
for _ in range(50):
    resp = requests.post("http://localhost:8085/predict", json={"prompt": prompt})
    char = resp.json().get("completion", "")
    print(char, end="", flush=True)
    prompt += char
    
    # We should only stop if it reaches the end of a story (double newline in TinyStories)
    if prompt.endswith('\n\n'):
        break
print()
