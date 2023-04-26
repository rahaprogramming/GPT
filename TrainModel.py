from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from torch.utils.data import DataLoader, TensorDataset

# Set the training parameters
batch_size = 32
num_epochs = 10
learning_rate = 3e-4

# Initialize the Wav2Vec2 model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Tokenize the preprocessed data
tokenized_data = []
for segment in preprocessed_data:
    tokens = processor(
        segment,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized_data.append(tokens)
tokenized_data = DataLoader(
    TensorDataset(torch.cat(tokenized_data)),
    batch_size=batch_size,
    shuffle=True
)

# Define the training loop
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id)
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    for i, batch in enumerate(tokenized_data):
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        input_lengths = torch.full(
            size=(logits.shape[0],), fill_value=logits.shape[1], dtype=torch.long
        )
        label_lengths = torch.sum(labels != processor.tokenizer.pad_token_id, dim=-1)
        loss = loss_fn(log_probs.transpose(0, 1), labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
