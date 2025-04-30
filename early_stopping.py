class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        Initialize early stopping
        
        Args:
            patience (int): Number of epochs to wait before stopping if no improvement
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Check if training should be stopped
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: True if training should be stopped, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return False 