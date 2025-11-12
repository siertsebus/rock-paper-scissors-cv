import sys
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
import os


def main() -> None:
    input_folder, output_folder = sys.argv[1], sys.argv[2]
    max_size = (100, 100)  # Resize to fit within 100x100
    quality = 100  # JPEG quality (0-100)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img: PILImage = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(max_size)
            output_path = os.path.join(output_folder, filename)
            img.save(output_path, "JPEG", quality=quality)


if __name__ == "__main__":
    main()
