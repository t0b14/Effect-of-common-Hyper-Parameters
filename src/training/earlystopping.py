import copy

class EarlyStopping():
    def __init__(self, patience=10, min_improvement=0, restore_best_weights=True):
        self.patience = patience
        self.min_improvement = min_improvement
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0 
        self.status = ""
        self.stop_early = False

    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)

        elif self.best_loss - val_loss > self.min_improvement:
            self.found_improvement(model, val_loss)

        elif self.best_loss - val_loss < self.min_improvement:
            self.stop_early = self.no_improvement(model)

        self.status = f"{self.counter}/{self.patience}"
        return self.stop_early
    
    def no_improvement(self, model):
        self.counter += 1
        if self.counter >= self.patience:
            self.status = f"Stopped on {self.counter}"
            if self.restore_best_weights:
                model.load_state_dict(self.best_model.state_dict())
            return True
        
        return False

    def found_improvement(self, model, val_loss):
        self.best_loss = val_loss
        self.counter = 0    
        self.best_model.load_state_dict(model.state_dict())