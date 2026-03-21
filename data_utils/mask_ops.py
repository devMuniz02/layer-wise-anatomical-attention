from pathlib import Path

from PIL import Image


def resize_mask(source_path: str, destination_path: str, image_size: int) -> None:
    image = Image.open(source_path).convert("L")
    resized = image.resize((image_size, image_size), resample=Image.NEAREST)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resized.save(destination, format="PNG")
