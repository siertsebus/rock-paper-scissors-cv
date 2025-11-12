from typing import Sequence
import matplotlib.pyplot as plt


def plot_losses(train_losses: Sequence[float], val_losses: Sequence[float]) -> None:
    """Plot training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
