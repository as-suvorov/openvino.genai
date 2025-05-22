import torchvision.transforms.v2
import openvino as ov
import torch
import torchvision
import numpy as np
from transformers import AutoProcessor
from phi4_mm_convertable_model import (
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


def get_vision_position_ids(
    pixel_values, patch_attention_mask, patch_size=14, num_patches_per_side=32
):
    batch_size = pixel_values.shape[0]
    max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
    max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    position_ids = torch.full(
        size=(
            batch_size,
            max_nb_patches_h * max_nb_patches_w,
        ),
        fill_value=0,
    )

    for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        nb_patches_h = p_attn_mask[:, 0].sum()
        nb_patches_w = p_attn_mask[0].sum()

        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (
            bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w
        ).flatten()
        position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
    return position_ids


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

        ov_output = self.preprocess_model(example_input)

        example_input = (
            torch.tensor(ov_output["input_image_embeds"]),
            torch.tensor(ov_output["image_attention_mask"]),
        )
        position_ids_model = PositionIdsModel()
        position_ids_model(*example_input)
        scripted_model = torch.jit.script(position_ids_model, [example_input])

        ov_model = ov.convert_model(
            scripted_model,
            example_input=example_input,
        )
        self.position_ids_model = ov.compile_model(ov_model, "CPU")

    def __call__(self, image):
        ov_output = self.target_sizes_model(image)
        new_size = ov_output["new_size"]
        padding_width = ov_output["padding_width"]
        padding_height = ov_output["padding_height"]
        attention_mask = ov_output["attention_mask"]

        preprocess_input = (
            image,
            attention_mask,
            new_size,
            padding_width,
            padding_height,
        )

        ov_output = self.preprocess_model(preprocess_input)
        input_image_embeds = ov_output["input_image_embeds"]
        image_attention_mask = ov_output["image_attention_mask"]
        image_height = ov_output["image_height"]
        image_width = ov_output["image_width"]
        num_img_tokens = ov_output["num_img_tokens"]

        patch_position_ids = self.position_ids_model(
            {
                "input_image_embeds.1": input_image_embeds,
                "image_attention_mask.1": image_attention_mask,
            }
        )["patch_position_ids"]

        return {
            "input_image_embeds": input_image_embeds,
            "image_attention_mask": image_attention_mask,
            "image_height": image_height,
            "image_width": image_width,
            "num_img_tokens": num_img_tokens,
            "patch_position_ids": patch_position_ids,
        }


ov_model = OVPreprocessModel()

for i, image in enumerate(images):
    transformers_start = time.perf_counter()
    preprocessed = processor.image_processor(tensor_to_pil(image[0]))
    transformers_time = time.perf_counter() - transformers_start

    original_embeds = preprocessed["input_image_embeds"]
    original_image_sizes = preprocessed["image_sizes"]
    original_attention_mask = preprocessed["image_attention_mask"]
    original_num_img_tokens = preprocessed["num_img_tokens"]

    original_patch_position_ids = get_vision_position_ids(
        original_embeds.flatten(0, 1),
        original_attention_mask.flatten(0, 1).to(dtype=bool),
    )

    ov_start = time.perf_counter()
    ov_outputs = ov_model(image)
    ov_time = time.perf_counter() - ov_start

    ov_embeds = ov_outputs["input_image_embeds"]
    ov_image_sizes = np.array((ov_outputs["image_height"], ov_outputs["image_width"]))
    ov_attention_mask = ov_outputs["image_attention_mask"]
    ov_num_img_tokens = ov_outputs["num_img_tokens"]
    ov_patch_position_ids = ov_outputs["patch_position_ids"]

    print(f"\n\n=============  image {i}  =============")
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
    print(
        f"patch position ids shapes: original -> {original_patch_position_ids.shape}, ov -> {ov_patch_position_ids.shape}"
    )
    print(f"embeds error original vs ov: {torch.max(original_embeds - ov_embeds)}")
    assert np.all(np.array(original_image_sizes == ov_image_sizes))
    assert np.all(np.array(original_attention_mask == ov_attention_mask))
    assert np.all(np.array(original_num_img_tokens == ov_num_img_tokens))
    assert np.all(np.array(original_patch_position_ids == ov_patch_position_ids))
    assert torch.max(original_embeds - ov_embeds) < 0.01
