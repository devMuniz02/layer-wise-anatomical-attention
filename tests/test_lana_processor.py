import numpy as np

from lana_radgen.image_processing_lana import LanaImageProcessor
from lana_radgen.model_card import build_dual_usage_section


def test_lana_image_processor_single_image_shape():
    processor = LanaImageProcessor(size={"height": 512, "width": 512})
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    batch = processor(images=image, return_tensors="np")
    assert batch["pixel_values"].shape == (1, 3, 512, 512)


def test_lana_image_processor_batch_shape():
    processor = LanaImageProcessor(size={"height": 512, "width": 512})
    images = [np.zeros((32, 32, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8)]
    batch = processor(images=images, return_tensors="np")
    assert batch["pixel_values"].shape == (2, 3, 512, 512)
    assert batch["pixel_values"].dtype == np.float32


def test_lana_image_processor_torch_dtype_is_float32():
    processor = LanaImageProcessor(size={"height": 512, "width": 512})
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    batch = processor(images=image, return_tensors="pt")
    assert str(batch["pixel_values"].dtype) == "torch.float32"


def test_model_card_usage_section_documents_both_paths():
    section = build_dual_usage_section("manu02/LAnA-v4")
    assert "Implementation 1: Standard Hugging Face loading" in section
    assert "Implementation 2: Snapshot download / legacy manual loading" in section
