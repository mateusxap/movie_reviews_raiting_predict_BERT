from django.db import models
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

class BertForSentimentAndRating(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSentimentAndRating, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Голова для классификации тональности
        self.classifier = nn.Linear(config.hidden_size, 2)

        # Голова для регрессии рейтинга
        self.regressor = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # CLS token representation
        pooled_output = self.dropout(pooled_output)

        sentiment_logits = self.classifier(pooled_output)
        rating_output = self.regressor(pooled_output)

        return sentiment_logits, rating_output