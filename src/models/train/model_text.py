from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM
import torch

# Initialize the model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking')

# Load the saved model state_dict
model_path = '/jf-training-home/NLP_Model/src/models/checkpoints/model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Example: Tokenize and classify a sample sentence
sample_sentence = "쟤는 연기도 진짜 못하네"
inputs = tokenizer(sample_sentence, return_tensors='pt', truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)

# The outputs now contain the logits for each class
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

print(f"Predicted Class: {predicted_class}")
print(f"Class Probabilities: {probabilities.numpy()}")
