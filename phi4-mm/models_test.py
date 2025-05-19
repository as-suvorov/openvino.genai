import torch
import openvino as ov
import numpy as np
from torchvision.transforms import v2
import functools
from PIL import Image
import requests
from io import BytesIO
from phi4_mm_convertable_model import Phi4MMConvertableModel
from phi4_mm_original_model import OriginalModel
import constants

to_pil_processor = v2.Compose([v2.ToPILImage()])
to_tensor_processor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


@functools.lru_cache
def get_test_images():
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    x2h_image = Image.open("tall_image.png")
    imgs_comb = np.vstack(
        [np.array(x2h_image), np.array(x2h_image), np.array(x2h_image)]
    )
    imgs_comb = Image.fromarray(imgs_comb).convert("RGB")
    images = [
        Image.open(BytesIO(requests.get(url).content)),
        Image.open("unnamed.jpg"),
        Image.open("final_results_chart.png"),
        imgs_comb,
    ]

    images = [
        torch.rand((3, 40000, 400)),
        torch.rand((3, 400, 400)),
        torch.rand((3, 200, 800)),
        torch.rand((3, 800, 200)),
        torch.rand((3, 530, 720)),
        torch.rand((3, 1245, 1334)),
    ]
    return [x for x in images]


def test_ov_model(
    label: str,
    pt_model,
    original_model,
    images: list[torch.Tensor],
    targetRatios: torch.Tensor,
):
    ov_model = ov.convert_model(pt_model, example_input=(images[0], targetRatios))

    ov_model_path = ".vscode/vlm/models/preprocess_image.xml"
    ov.save_model(ov_model, ov_model_path)

    ov_model = ov.compile_model(ov_model, "CPU")

    print(f"====== {label} ====")

    for i, image in enumerate(images):
        ov_outputs = ov_model((image, targetRatios))
        ov_embeds = ov_outputs["input_image_embeds"]
        ov_image_sizes = np.array(
            (ov_outputs["image_height"], ov_outputs["image_width"])
        )
        ov_attention_mask = ov_outputs["image_attention_mask"]
        ov_num_img_tokens = ov_outputs["num_img_tokens"]

        pt_output = pt_model(image, targetRatios)
        pt_embeds = pt_output["input_image_embeds"]
        pt_image_sizes = np.array((pt_output["image_height"], pt_output["image_width"]))
        pt_attention_mask = pt_output["image_attention_mask"]
        pt_num_img_tokens = pt_output["num_img_tokens"]

        pil_image = to_pil_processor(image)
        original_outputs = original_model([pil_image])
        original_embeds = original_outputs["input_image_embeds"]
        original_image_sizes = original_outputs["image_sizes"]
        original_attention_mask = original_outputs["image_attention_mask"]
        original_num_img_tokens = original_outputs["num_img_tokens"]

        print(f"\n\n=============  image {i}  =============")

        print(
            f"image_sizes shapes: original -> {original_image_sizes} pt -> {pt_image_sizes}, ov -> {ov_image_sizes}"
        )
        assert np.all(np.array(pt_image_sizes == ov_image_sizes))
        assert np.all(np.array(original_image_sizes == ov_image_sizes))

        print(
            f"attention_mask shapes: original -> {original_attention_mask.shape} pt -> {pt_attention_mask.shape}, ov -> {ov_attention_mask.shape}"
        )
        assert np.all(np.array(pt_attention_mask == ov_attention_mask))
        assert np.all(np.array(original_attention_mask == ov_attention_mask))

        print(
            f"num_img_tokens shapes: original -> {original_num_img_tokens} pt -> {pt_num_img_tokens}, ov -> {ov_num_img_tokens}"
        )
        assert np.all(np.array(pt_num_img_tokens == ov_num_img_tokens))
        assert np.all(np.array(original_num_img_tokens == ov_num_img_tokens))

        print(
            f"embeds shapes: original -> {original_embeds.shape} pt -> {pt_embeds.shape}, ov -> {ov_embeds.shape}"
        )

        print(f"embeds error original vs pt: {torch.max(original_embeds - pt_embeds)}")
        print(f"embeds error pt vs ov: {torch.max(pt_embeds - ov_embeds)}")
        print(f"embeds error original vs ov: {torch.max(original_embeds - ov_embeds)}")


test_images = get_test_images()

pil_images = [to_pil_processor(x) for x in test_images]
original_model = OriginalModel()
original_output = original_model([pil_images[0]])


phi4_image_preprocessing_model = Phi4MMConvertableModel()

# scripted_model = torch.jit.script(
#     test_model, [pt_images, torch.tensor(utils.SORTED_TARGET_RATIOS)]
# )


ov_model = ov.convert_model(
    phi4_image_preprocessing_model,
    example_input=(get_test_images()[0], torch.tensor(constants.SORTED_TARGET_RATIOS)),
)

ov_model_path = ".vscode/vlm/models/preprocess_image.xml"
ov.save_model(ov_model, ov_model_path)

ov_model = ov.compile_model(ov_model, "CPU")

# from openvino_devtools import ov2py

# result = ov2py.ov2py(ov.Core().read_model(ov_model_path))

# print(result)

# test_ov_model(
#     "covertable pytorch vs ov",
#     phi4_image_preprocessing_model,
#     original_model,
#     to_tensor_processor(get_test_images()),
#     torch.tensor(constants.SORTED_TARGET_RATIOS),
# )
