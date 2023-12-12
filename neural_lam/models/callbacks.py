import torch
import pytorch_lightning as pl

# Define a custom callback to store predictions during testing
class PredictionCallback(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        predictions = torch.cat(pl_module.test_predictions, dim=0)
        true_labels = torch.cat(pl_module.test_true_labels, dim=0)
        test_loss = torch.tensor(pl_module.test_loss).mean().item()

        print(f'Test Loss: {test_loss}')
        return predictions