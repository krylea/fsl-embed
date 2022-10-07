import torch
from transformers import Trainer


class VQTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs)
        
        loss += model.bert.embeddings.word_embeddings.symbol_loss_buffer

        if return_outputs:
            return loss, outputs
        else:
            return loss


