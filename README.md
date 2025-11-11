# rock-paper-scissors-cv
A super short toy project on training a computer vision model to classify hand gestures as either Rock, Paper, or Scissors.

## Data preprocessing
Phone pictures are rather high-quality and large for a toy project. Downscale them by using `preprocess_raw_pictures.py`:
```sh
python preprocess_raw_pictures.py ./rps_dataset_raw ./rps_dataset
```