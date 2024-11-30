import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import numpy as np
import json

# Load dataset
file_path = "data/sofmattress_train.csv"
data = pd.read_csv(file_path)
texts = data['sentence'].tolist()
labels = data['label'].tolist()

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(text_list, max_length=128):
    encoded = tokenizer(
        text_list,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded

encoded_data = tokenize_texts(texts)
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']

# Label encoding
label_map = {label: idx for idx, label in enumerate(set(labels))}
reverse_label_map = {idx: label for label, idx in label_map.items()}
labels_encoded = torch.tensor([label_map[label] for label in labels])

# Split data
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids, labels_encoded, test_size=0.2, random_state=42
)
train_masks, val_masks, _, _ = train_test_split(
    attention_masks, labels_encoded, test_size=0.2, random_state=42
)

batch_size = 32
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Load BERT model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
model.to(device)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(labels_encoded.numpy()), y=labels_encoded.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = CrossEntropyLoss(weight=class_weights)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

# Training loop
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training"):
        batch_input_ids, batch_attention_mask, batch_labels = (
            batch[0].to(device), batch[1].to(device), batch[2].to(device)
        )
        model.zero_grad()
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, batch_labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Average Training Loss = {total_loss / len(train_dataloader)}")

    # Validation loop
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} Validating"):
        batch_input_ids, batch_attention_mask, batch_labels = (
            batch[0].to(device), batch[1].to(device), batch[2].to(device)
        )
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, batch_labels)
        val_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())

    print(f"Epoch {epoch + 1}: Average Validation Loss = {val_loss / len(val_dataloader)}")
    print(classification_report(all_labels, all_preds, target_names=list(label_map.keys())))

# Save model and tokenizer
save_directory = "bert_intent_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Save reverse label map
with open(f"{save_directory}/label_map.json", "w") as f:
    json.dump(reverse_label_map, f)

print("Training complete and model saved.")
