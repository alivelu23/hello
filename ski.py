from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.squeeze(0)

def calculate_inconsistency(source_text, summary):
    source_embeddings = get_embeddings(source_text)
    summary_embeddings = get_embeddings(summary)

    num_inconsistencies = 0
    for summary_embedding in summary_embeddings:
        similarities = torch.matmul(source_embeddings, summary_embedding)
        max_similarity_index = torch.argmax(similarities)
        
        # Check if tokens match (assuming tokenization is aligned)
        if tokenizer.decode([inputs['input_ids'][0][max_similarity_index]]) != tokenizer.decode([inputs['input_ids'][0][summary_embeddings.index(summary_embedding)]]):
            num_inconsistencies += 1

    return num_inconsistencies

# Example usage
source_text = "Your source document text here."
summary = "Your summary text here."
inconsistencies = calculate_inconsistency(source_text, summary)
print(f"Number of inconsistencies: {inconsistencies}")
