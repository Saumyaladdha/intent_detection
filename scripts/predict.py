import json
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load model and tokenizer
model_directory = "bert_intent_model"
model = BertForSequenceClassification.from_pretrained(model_directory)
tokenizer = BertTokenizer.from_pretrained(model_directory)
model.eval()

# Load label map
with open(f"{model_directory}/label_map.json", "r") as f:
    reverse_label_map = json.load(f)

# Prediction function
def predict_intent(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return reverse_label_map[str(predicted_label)]

# Test cases
if __name__ == "__main__":
    test_cases = [
        "How do I return a damaged item?",
        "Why hasn’t my package arrived yet?",
        "Can I pay for my order upon delivery?",
        "I'm interested in learning more about your services.",
        "Could you schedule a call with a support agent for tomorrow?"
    ]

    for sentence in test_cases:
        print(f"Sentence: {sentence} | Predicted Intent: {predict_intent(sentence)}")
