import torch
from src.training.training_abstractbaseclass import ABCTrainingModule

class RNNTrainingModule1(ABCTrainingModule):
    def __init__(self, model, optimizer, params) -> None:
        super().__init__(model, optimizer, params)
        self.loss = torch.nn.MSELoss()

    def compute_loss(self, inputs, labels):
        out = self.model(inputs)
        return out, self.loss(out, labels)