import torch
from torchmetrics import BinaryAccuracy

from src.training.training_abstractbaseclass import ABCTrainingModule


class RNNTrainingModule1(ABCTrainingModule):
    def __init__(self, model, optimizer, params) -> None:
        super().__init__(model, optimizer, params)
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = BinaryAccuracy(threshold = 0.5).to(
            self.device
        )

    def compute_loss(self, inputs, labels):
        out = self.model(inputs)
        return out, self.loss(out, labels)

    def compute_test_error(self, predictions, labels):
        return (1 - self.accuracy(predictions, labels)).item()

    def compute_metrics(self, predictions, labels):
        return {"Error": self.compute_test_error(predictions, labels)}