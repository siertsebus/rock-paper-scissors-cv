import torchvision

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

    dataset = RockPaperScissorsDataset(img_dir="rps_dataset")

    print("chicken")


if __name__ == "__main__":
    main()
