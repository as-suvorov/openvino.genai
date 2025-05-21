import torchvision.transforms.v2
from transformers import AutoProcessor
from phi4_mm_convertable_model import Phi4MMConvertableModel
from optimum.intel import OVModelForVisualCausalLM
import openvino as ov
import torch
import torchvision
import numpy as np
import soundfile
from transformers import AutoProcessor, TextStreamer
from io import BytesIO

import requests
from PIL import Image


tensor_to_pil = torchvision.transforms.v2.ToPILImage()


model_id = "microsoft/Phi-4-multimodal-instruct"

images = [
    torch.randint(256, size=(1, 3, 400, 400), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 200, 800), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 800, 200), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 530, 720), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 1245, 1334), dtype=torch.uint8),
    torch.randint(256, size=(1, 3, 1200, 768), dtype=torch.uint8),
]

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
ov_model = OVModelForVisualCausalLM.from_pretrained(
    ".vscode/models/phi4-mm", trust_remote_code=True
)
# ov_model.save_pretrained(".vscode/models/phi4-mm")
url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image = Image.open(BytesIO(requests.get(url).content))
inputs = ov_model.preprocess_inputs(
    text="What is unusual on this picture?",
    image=[tensor_to_pil(images[1][0])],
    processor=processor,
)
# Image-Text

print("Question:\nWhat is unusual on this picture?")
print("Answer:")

generate_ids = ov_model.generate(
    **inputs,
    max_new_tokens=100,
    streamer=TextStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    ),
)
