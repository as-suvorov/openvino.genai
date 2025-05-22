import torchvision.transforms.v2
import openvino as ov
import torch
import torchvision
import numpy as np
from transformers import AutoProcessor
from phi4_mm_convertable_model import (
    Phi4MMConvertableModel,
    PositionIdsModel,
    TargetSizesModel,
    PreProcessModel,
)
import time


def tensor_to_pil(value: torch.Tensor):
    # hwc -> chw
    value = value.permute(2, 0, 1).contiguous()
    return torchvision.transforms.v2.ToPILImage()(value)


model_id = "microsoft/Phi-4-multimodal-instruct"

images = [
    torch.randint(256, size=(1, 400, 400, 3), dtype=torch.uint8),
    torch.randint(256, size=(1, 200, 800, 3), dtype=torch.uint8),
    torch.randint(256, size=(1, 800, 200, 3), dtype=torch.uint8),
    torch.randint(256, size=(1, 530, 720, 3), dtype=torch.uint8),
    torch.randint(256, size=(1, 1245, 1334, 3), dtype=torch.uint8),
    torch.randint(256, size=(1, 1200, 768, 3), dtype=torch.uint8),
    torch.randint(256, size=(1, 2700, 2500, 3), dtype=torch.uint8),
    torch.randint(256, size=(1, 40000, 400, 3), dtype=torch.uint8),
]

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


class OVPreprocessModel:
    def __init__(self):
        target_sizes_model = TargetSizesModel()
        scripted_model = torch.jit.script(target_sizes_model, [images[0]])

        ov_target_sizes_model = ov.convert_model(
            scripted_model, example_input=images[0]
        )
        self.target_sizes_model = ov.compile_model(ov_target_sizes_model, "CPU")

        ov_output = self.target_sizes_model(images[0])
        ov_new_size = ov_output["new_size"]
        ov_padding_width = ov_output["padding_width"]
        ov_padding_height = ov_output["padding_height"]
        ov_attention_mask = ov_output["attention_mask"]

        example_input = (
            images[0],
            torch.tensor(ov_attention_mask),
            torch.tensor(ov_new_size),
            torch.tensor(ov_padding_width),
            torch.tensor(ov_padding_height),
        )

        preprocess_model = PreProcessModel()

        ov_preprocess_model = ov.convert_model(
            preprocess_model, example_input=example_input
        )
        self.preprocess_model = ov.compile_model(ov_preprocess_model, "CPU")

    def __call__(self, image):
        ov_output = self.target_sizes_model(image)
        new_size = ov_output["new_size"]
        padding_width = ov_output["padding_width"]
        padding_height = ov_output["padding_height"]
        attention_mask = ov_output["attention_mask"]

        example_input = (
            image,
            attention_mask,
            new_size,
            padding_width,
            padding_height,
        )

        return self.preprocess_model(example_input)


ov_model = OVPreprocessModel()

for i, image in enumerate(images):
    transformers_start = time.perf_counter()
    preprocessed = processor.image_processor(tensor_to_pil(image[0]))
    transformers_time = time.perf_counter() - transformers_start

    original_embeds = preprocessed["input_image_embeds"]
    original_image_sizes = preprocessed["image_sizes"]
    original_attention_mask = preprocessed["image_attention_mask"]
    original_num_img_tokens = preprocessed["num_img_tokens"]

    ov_start = time.perf_counter()
    ov_outputs = ov_model(image)
    ov_time = time.perf_counter() - ov_start

    ov_embeds = ov_outputs["input_image_embeds"]
    ov_image_sizes = np.array((ov_outputs["image_height"], ov_outputs["image_width"]))
    ov_attention_mask = ov_outputs["image_attention_mask"]
    ov_num_img_tokens = ov_outputs["num_img_tokens"]

    print(f"image {i}")
    print(f"transformers time: {transformers_time:.4f}s, ov time: {ov_time:.4f}s")
    print(
        f"embeds shapes: original -> {original_embeds.shape}, ov -> {ov_embeds.shape}"
    )
    print(f"image sizes: original -> {original_image_sizes}, ov -> {ov_image_sizes}")
    print(
        f"atteintion mask: original -> {original_attention_mask.shape}, ov -> {ov_attention_mask.shape}"
    )
    print(
        f"num img tokens: original -> {original_num_img_tokens}, ov -> {ov_num_img_tokens}"
    )
    print(f"embeds error original vs ov: {torch.max(original_embeds - ov_embeds)}")
    assert np.all(np.array(original_image_sizes == ov_image_sizes))
    assert np.all(np.array(original_attention_mask == ov_attention_mask))
    assert np.all(np.array(original_num_img_tokens == ov_num_img_tokens))
    assert torch.max(original_embeds - ov_embeds) < 0.01
