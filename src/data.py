import torch
from torch.utils.data import Dataset

class ChatbotDataset(Dataset):
    def __init__(self, tokenizer, dialogues, max_length=100):
        self.tokenizer = tokenizer
        self.dialogues = dialogues
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        input_text, target_text = dialogue['input'], dialogue['target']
        input_tokens = self.tokenizer.encode(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        target_tokens = self.tokenizer.encode(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        return input_tokens.squeeze(0), target_tokens.squeeze(0)

def collate_fn(batch):
    input_tokens, target_tokens = zip(*batch)
    input_tokens = torch.stack(input_tokens)
    target_tokens = torch.stack(target_tokens)
    return input_tokens, target_tokens

