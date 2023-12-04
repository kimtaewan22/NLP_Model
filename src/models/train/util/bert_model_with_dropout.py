import torch.nn as nn
from transformers import BertForSequenceClassification, AdamW

class BertModelWithDropout(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(BertModelWithDropout, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_ids, labels=None):
        outputs = self.bert(input_ids, labels=labels)
        logits = self.dropout(outputs.logits)  # Apply dropout to the logits
        return logits