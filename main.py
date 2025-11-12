import copy

import torchvision
from torchvision.transforms import v2 as transforms
import torch
from tqdm import tqdm

from rps_dataset import RockPaperScissorsDataset


def define_transform() -> transforms.Transform:
    transform_crop = transforms.RandomResizedCrop(
        size=(100, 100), scale=(0.5, 1.0), ratio=(1, 1)
    )
    transform_choice = transforms.RandomChoice(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomGrayscale(),
        ]
    )
    return transforms.Compose([transform_crop, transform_choice, transform_choice])


def main() -> None:
    pretrained_model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )

    # Load dataset and split
    transform = define_transform()
    dataset = RockPaperScissorsDataset(img_dir="rps_dataset", transform=transform)
    train_set, val_set, test_set = dataset.balanced_split([0.7, 0.15, 0.15])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

    # Create a copy of the pretrained model and set it up for 3-class classification
    model: torchvision.models.ResNet = copy.deepcopy(pretrained_model)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=3)
    model.fc.reset_parameters()

    # freeze all but the last set of conv layers:
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    loss = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    n_epochs = 50
    epoch_iter = tqdm(range(n_epochs), desc="Epochs")
    for _ in epoch_iter:
        # train loop
        model.train()
        train_loss_total = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            train_loss = loss(y_pred, y)
            train_loss.backward()
            optimizer.step()
            train_loss_total += train_loss.item()

        # validation loop
        model.eval()
        val_loss_total = 0.0
        for x, y in val_loader:
            y_pred = model(x)
            val_loss = loss(y_pred, y)
            val_loss_total += val_loss.item()

        epoch_iter.set_postfix(
            loss=train_loss_total / len(train_set),
            val_loss=val_loss_total / len(val_set),
        )

    # Evaluate both models on the test set
    model.eval()
    dataset.transform = None  # Disable data augmentation for evaluation
    with torch.no_grad():
        test_loss_total = 0.0
        test_correct_total = 0
        for x, y in test_loader:
            y_pred = model(x)
            loss_value = loss(y_pred, y)
            top_class = y_pred.argmax(dim=1)
            test_loss_total += loss_value.item()
            test_correct_total += (
                (top_class == torch.tensor(y, dtype=torch.int64)).sum().item()
            )

    print(f"Final test loss: {test_loss_total / len(test_set)}")
    print(f"Final test accuracy: {test_correct_total / len(test_set)}")


if __name__ == "__main__":
    main()
