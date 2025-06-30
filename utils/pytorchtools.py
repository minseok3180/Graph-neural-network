import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, verbose=False, delta=0, checkpoint_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = 0
        self.delta = delta
        self.cleared = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score <= self.best_score - self.delta:
            self.counter += 1
            print('\033[1;31mEarlyStopping counter: {} out of {}\033[0m'
                  .format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print('\033[1;31mTest score increased ({:.6f} --> {:.6f}).\033[0m'
                      .format(self.best_score, score))
            self.counter = 0
            self.save_checkpoint(model)
            self.best_score = score

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self, model):
        if os.path.exists(self.checkpoint_path):
            model.load_state_dict(torch.load(self.checkpoint_path))
            print(f"\033[1;34mLoaded best model from {self.checkpoint_path}\033[0m")
        else:
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")