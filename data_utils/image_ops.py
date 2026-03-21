from pathlib import Path

from PIL import Image


def resize_rgb_image(source_path: str, destination_path: str, image_size: int) -> None:
    image = Image.open(source_path).convert("RGB")
    resized = image.resize((image_size, image_size), resample=Image.BICUBIC)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resized.save(destination, format="PNG")
