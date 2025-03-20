import torch
from transformers import LlamaTokenizer
from lora_model import LLaMAWithLoRA


def train(model, tokenizer, device, epochs=1):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Dummy training data
    texts = ["Hello, how are you?", "What is your name?", "Tell me a joke."]

    for epoch in range(epochs):
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            optimizer.zero_grad()

            outputs = model(**inputs)

            # Dummy loss calculation
            loss = outputs.sum()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    model_name = "llama-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LLaMAWithLoRA(model_name).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    train(model, tokenizer, device, epochs=3)