import torchvision.transforms.v2
from transformers import AutoProcessor
from phi4_mm_convertable_model import Phi4MMConvertableModel
import openvino as ov
import torch
import torchvision
import numpy as np


tensor_to_pil = torchvision.transforms.v2.ToPILImage()


model_id = "microsoft/Phi-4-multimodal-instruct"

images = [
    torch.rand((3, 400, 400)),
    torch.rand((3, 200, 800)),
    torch.rand((3, 800, 200)),
    torch.rand((3, 530, 720)),
    torch.rand((3, 1245, 1334)),
    torch.rand((3, 1200, 768)),
]

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


phi4_image_preprocessing_model = Phi4MMConvertableModel()

ov_model = ov.convert_model(
    phi4_image_preprocessing_model,
    example_input=images[0],
)
ov_compiled_model = ov.compile_model(ov_model, "CPU")


for i, image in enumerate(images):
    preprocessed = processor.image_processor(tensor_to_pil(image))
    original_embeds = preprocessed["input_image_embeds"]
    original_image_sizes = preprocessed["image_sizes"]
    original_attention_mask = preprocessed["image_attention_mask"]
    original_num_img_tokens = preprocessed["num_img_tokens"]

    ov_outputs = ov_compiled_model(image)
    ov_embeds = ov_outputs["input_image_embeds"]
    ov_image_sizes = np.array((ov_outputs["image_height"], ov_outputs["image_width"]))
    ov_attention_mask = ov_outputs["image_attention_mask"]
    ov_num_img_tokens = ov_outputs["num_img_tokens"]

    print(f"image {i}")
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
