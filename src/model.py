import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import CrossEntropyLoss

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

    def train(self, train_dataloader, val_dataloader, epochs, learning_rate, log_interval):
    self.model.train()

    optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    loss_function = CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_idx, (input_tokens, target_tokens) in enumerate(train_dataloader):
            input_tokens, target_tokens = input_tokens.to(self.device), target_tokens.to(self.device)
            optimizer.zero_grad()

            outputs = self.model(input_tokens, labels=target_tokens)
            loss = outputs[0]

            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item()}")

        # Evaluate on the validation dataset
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (input_tokens, target_tokens) in enumerate(val_dataloader):
                input_tokens, target_tokens = input_tokens.to(self.device), target_tokens.to(self.device)
                outputs = self.model(input_tokens, labels=target_tokens)
                loss = outputs[0]
                total_loss += loss.item()

        avg_loss = total_loss / len(val_dataloader)
        print(f"Validation Loss (Epoch {epoch}): {avg_loss}")

# Add this function to the ChatBotModel class

