from transformers import BertTokenizer

def tokenize_texts(texts, tokenizer_path, max_length=128):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
    )
    return encoded["input_ids"], encoded["attention_mask"]
