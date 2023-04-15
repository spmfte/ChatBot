import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ChatBotModel:
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

    def generate_response(self, input_text, max_length=100):
        input_tokens = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        response_tokens = self.model.generate(input_tokens, max_length=max_length, num_return_sequences=1)
        response_text = self.tokenizer.decode(response_tokens[0], skip_special_tokens=True)
        return response_text

