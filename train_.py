import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset, random_split
from accelerate import Accelerator
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import login
import math
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoding.input_ids.squeeze(), encoding.attention_mask.squeeze()


def train(model, train_dataloader, val_dataloader, optimizer, accelerator):
    model.train()
    loss_history = []
    val_loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask in train_dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        loss_history.append(avg_loss)
        perplexity = math.exp(avg_loss)
        print(f'Epoch {epoch + 1}/{epochs} | Training Loss: {avg_loss} | Perplexity: {perplexity}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask in val_dataloader:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                val_loss += outputs.loss.item()
        val_loss /= len(val_dataloader)
        val_loss_history.append(val_loss)
        print(f'Epoch {epoch + 1}/{epochs} | Validation Loss: {val_loss}')
        model.train()

        model.save_pretrained(f"./lora-llama-epoch-{epoch + 1}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.show()


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_model_for_inference(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


if __name__ == "__main__":
    hf_token = input("Enter your Hugging Face token: ")
    login(hf_token)

    accelerator = Accelerator()

    model_id = "huggyllama/llama-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_token).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)

    data_path = "./training_data.txt"
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
    else:
        data = [
            "Hello, I'm Dawood MD. Welcome to my GitHub!",
            "The quick brown fox jumps over the lazy dog.",
            "Once upon a time, in a land far, far away..."
        ]

    dataset = CustomDataset(data, tokenizer, max_length=128)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, pin_memory=True)

    model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(
        model, train_dataloader, val_dataloader, torch.optim.AdamW(model.parameters(), lr=1e-4)
    )

    epochs = 3

    train(model, train_dataloader, val_dataloader, optimizer, accelerator)

    model.save_pretrained("./lora-llama-final")
    tokenizer.save_pretrained("./lora-llama-final")

    model, tokenizer = load_model_for_inference("./lora-llama-final")

    while True:
        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        generated_text = generate_text(model, tokenizer, prompt, max_length=150, temperature=0.8, top_k=40, top_p=0.9)
        print("Generated Text:\n", generated_text)