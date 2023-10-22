import torch
from src.training.training_abstractbaseclass import ABCTrainingModule

class RNNTrainingModule1(ABCTrainingModule):
    def __init__(self, model, optimizer, params) -> None:
        super().__init__(model, optimizer, params)
        self.loss = torch.nn.MSELoss(reduction='sum')

    def compute_loss(self, inputs, labels):
        out = self.model(inputs)
        return out, self.loss(out, labels)
    
    def compute_metrics(self, model_predictions, labels): # TODO what metric?
        metric = {"test_loss": [(self.loss(model_predictions[i], labels[i])).item() for i in range(len(model_predictions))]}
        return metric