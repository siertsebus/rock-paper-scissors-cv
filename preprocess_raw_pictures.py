import sys
from PIL import Image
import os


def main() -> None:
    input_folder, output_folder = sys.argv[1], sys.argv[2]
    max_size = (100, 100)  # Resize to fit within 100x100
    quality = 100  # JPEG quality (0-100)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img.thumbnail(max_size)
            output_path = os.path.join(output_folder, filename)
            img.save(output_path, 'JPEG', quality=quality)


if __name__ == "__main__":
    main()
