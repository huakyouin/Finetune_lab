import torch
import torch.nn as nn
from transformers import BertModel


class BertForSequenceClassification(nn.Module):
    def __init__(self, bert, output_dim, freeze):
        super().__init__()
        self.bert = bert
        hidden_dim = bert.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, ids, attention_mask=None):
        # ids = [batch size, seq len]
        output = self.bert(ids, output_attentions=False, attention_mask = attention_mask)
        hidden = output.last_hidden_state
        # hidden = [batch size, seq len, hidden dim]
        # attention = output.attentions[-1]
        # attention = [batch size, n heads, seq len, seq len]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        # prediction = [batch size, output dim]
        return prediction