import copy

import torchvision
import torch
from tqdm import tqdm

from rps_dataset import RockPaperScissorsDataset

# Plan
# 1. Load pretrained model
# 2. Load in dataset, split in train/test
# 3. Create a copy of the model, replace last layer with 3 output neurons
# 4. Train the model on the training set (with most layers frozen)
# 5. Evaluate both models on the test set


def main() -> None:
    pretrained_model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )

    # Load dataset and split
    dataset = RockPaperScissorsDataset(img_dir="rps_dataset")
    train_set, val_set, test_set = dataset.balanced_split([0.7, 0.15, 0.15])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

    # Create a copy of the pretrained model and set it up for 3-class classification
    model: torchvision.models.ResNet = copy.deepcopy(pretrained_model)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=3)

    # freeze all but the last set of conv layers:
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    epoch_iter = tqdm(range(10), desc="Epochs")
    for _ in epoch_iter:
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss_value = loss(y_pred, y)
            loss_value.backward()
            optimizer.step()
        epoch_iter.set_postfix(loss=loss_value.item())

    print("chicken")

    # Evaluate both models on the test set
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for x, y in test_loader:
            y_pred = model(x)
            loss_value = loss(y_pred, y)
            total_loss += loss_value.item()
        print(f"Test loss: {total_loss / len(test_loader)}")


if __name__ == "__main__":
    main()
