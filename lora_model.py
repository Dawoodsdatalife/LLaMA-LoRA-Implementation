import torch
import torch.nn as nn
from transformers import LlamaModel

class LoRALayer(nn.Module):
    def __init__(self, input_dim, rank):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, input_dim))

    def forward(self, x):
        return x + (x @ self.lora_A @ self.lora_B)


class LLaMAWithLoRA(nn.Module):
    def __init__(self, model_name, lora_rank=4):
        super(LLaMAWithLoRA, self).__init__()
        self.model = LlamaModel.from_pretrained(model_name)
        self.lora_layers = nn.ModuleList([LoRALayer(self.model.config.hidden_size, lora_rank) for _ in range(self.model.config.num_hidden_layers)])

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        for lora_layer in self.lora_layers:
            hidden_states = lora_layer(hidden_states)

        return hidden_states
