from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel,BertForSequenceClassification

class BertEmbeddingWrapper(BertEmbeddings):
    def __init__(self, config, symbolic_embeds):
        super().__init__(config)
        self.word_embeddings = symbolic_embeds

class BertModelWrapper(BertModel):
    def __init__(self, config, symbolic_embeds, **kwargs):
        super().__init__(config, **kwargs)
        self.embeddings = BertEmbeddingWrapper(config, symbolic_embeds)


class BertForSequenceClassificationWrapper(BertForSequenceClassification):
    def __init__(self, config, symbolic_embeds):
        super().__init__(config)
        self.bert = BertModelWrapper(config, symbolic_embeds)

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad_(False)
        
        for param in self.bert.embeddings.parameters():
            param.requires_grad_(True)