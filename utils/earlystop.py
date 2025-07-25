class EarlyStopping:
    """
    """
    def __init__(self, patience=7, delta=0):
        """
        Args:
            - patience (int): How long to wait after last time validation loss improved
                Default: 7.
            - detal (float): Minimum change in the monitored quantity to quanlify as an improvements.
                Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.save = False
        self.delta = delta
    
    def __call__(self, mean_val_loss, best_val_loss):
        if best_val_loss is None:
            self.save = True
        elif mean_val_loss >= best_val_loss + self.delta:
            self.counter += 1
            self.save = False
            if self.counter >= self.patience:
                self.early_stop = True
            else:
                self.save = True
                self.counter = 0 