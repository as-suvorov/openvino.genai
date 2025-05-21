import torchvision.transforms.v2
from transformers import AutoProcessor
from phi4_mm_convertable_model import PositionIdsModel
import openvino as ov
import torch
import torchvision
import numpy as np
from transformers import AutoProcessor
import time

tensor_to_pil = torchvision.transforms.v2.ToPILImage()


model_id = "microsoft/Phi-4-multimodal-instruct"

images = [
    torch.randint(256, size=(1, 3, 2700, 2500), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 40000, 400), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 400, 400), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 200, 800), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 800, 200), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 530, 720), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 1245, 1334), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 1200, 768), dtype=torch.uint8),
]


def get_position_ids_example_inputs(processor):
    pairs = []
    for image in images:
        outputs = processor.image_processor(tensor_to_pil(image[0]))
        embeds = outputs["input_image_embeds"]
        attention_mask = outputs["image_attention_mask"]
        pairs.append((embeds, attention_mask))
    return pairs


processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

example_inputs = get_position_ids_example_inputs(processor)
pt_model = PositionIdsModel()

scripted_model = torch.jit.script(pt_model, example_inputs=example_inputs)

ov_model = ov.convert_model(
    scripted_model,
    example_input=example_inputs[0],
)
ov_model = ov.compile_model(ov_model, "CPU")

for i, image in enumerate(images):
    preprocessed = processor.image_processor(tensor_to_pil(image[0]))
    input_image_embeds = preprocessed["input_image_embeds"]
    image_attention_mask = preprocessed["image_attention_mask"]

    pt_start = time.perf_counter()
    pt_position_ids = pt_model(input_image_embeds, image_attention_mask)
    pt_time = time.perf_counter() - pt_start

    ov_start = time.perf_counter()
    ov_position_ids = ov_model(
        {
            "input_image_embeds.1": input_image_embeds,
            "image_attention_mask.1": image_attention_mask,
        }
    )[0]
    ov_time = time.perf_counter() - ov_start

    print(f"\n\n=============  image {i}  =============")
    print(f"pt time: {pt_time:.4f} ov time: {ov_time:.4f}")
    print(
        f"position_ids shapes: pt -> {pt_position_ids.shape}, ov -> {ov_position_ids.shape}"
    )
    assert np.all(np.array(pt_position_ids) == np.array(ov_position_ids))
