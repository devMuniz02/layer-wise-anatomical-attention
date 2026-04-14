from __future__ import annotations

from typing import Any

import numpy as np
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb, normalize, resize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import TensorType


class LanaImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = get_size_dict(size or {"height": 512, "width": 512})
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]
        self.do_convert_rgb = do_convert_rgb

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs: Any,
    ) -> BatchFeature:
        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError("LanaImageProcessor expected a PIL image, numpy array, torch tensor, or a list of images.")

        pixel_values = []
        for image in images:
            if self.do_convert_rgb:
                image = convert_to_rgb(image)
            array = to_numpy_array(image).astype(np.float32)
            input_data_format = infer_channel_dimension_format(array)
            if self.do_resize:
                array = resize(
                    image=array,
                    size=(self.size["height"], self.size["width"]),
                    resample=self.resample,
                    input_data_format=input_data_format,
                )
                input_data_format = infer_channel_dimension_format(array)
            if self.do_rescale:
                array = array * self.rescale_factor
            if self.do_normalize:
                array = normalize(
                    array,
                    mean=self.image_mean,
                    std=self.image_std,
                    input_data_format=input_data_format,
                )
            array = to_channel_dimension_format(array, data_format, input_channel_dim=input_data_format)
            array = np.asarray(array, dtype=np.float32)
            pixel_values.append(array)

        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)
