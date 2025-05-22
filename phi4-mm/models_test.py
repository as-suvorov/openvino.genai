import torch
import openvino as ov
import numpy as np
from torchvision.transforms import v2
import functools
from PIL import Image
import requests
from io import BytesIO
from phi4_mm_convertable_model import (
    Phi4MMConvertableModel,
    PositionIdsModel,
    TargetSizesModel,
    PreProcessModel,
)
from phi4_mm_original_model import OriginalModel
import constants
import time


def to_pil_processor(value: torch.Tensor):
    # hwc -> chw
    value = value.permute(2, 0, 1).contiguous()
    return v2.Compose([v2.ToPILImage()])(value)


to_tensor_processor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


@functools.lru_cache
def get_test_images():
    images = [
        torch.randint(256, size=(1, 2700, 2500, 3), dtype=torch.uint8),
        torch.randint(256, size=(1, 40000, 400, 3), dtype=torch.uint8),
        torch.randint(256, size=(1, 400, 400, 3), dtype=torch.uint8),
        torch.randint(256, size=(1, 200, 800, 3), dtype=torch.uint8),
        torch.randint(256, size=(1, 800, 200, 3), dtype=torch.uint8),
        torch.randint(256, size=(1, 530, 720, 3), dtype=torch.uint8),
        torch.randint(256, size=(1, 1245, 1334, 3), dtype=torch.uint8),
        torch.randint(256, size=(1, 1200, 768, 3), dtype=torch.uint8),
    ]
    return [x for x in images]


def test_ov_model(
    label: str,
    pt_model,
    original_model,
    images: list[torch.Tensor],
):
    ov_model = ov.convert_model(pt_model, example_input=images[1])

    ov_model_path = ".vscode/vlm/models/preprocess_image.xml"
    ov.save_model(ov_model, ov_model_path)

    ov_model = ov.compile_model(ov_model, "CPU")

    print(f"====== {label} ====")

    for i, image in enumerate(images):
        ov_start = time.time()
        ov_outputs = ov_model(image)
        ov_time = time.time() - ov_start
        ov_embeds = ov_outputs["input_image_embeds"]
        ov_image_sizes = np.array(
            (ov_outputs["image_height"], ov_outputs["image_width"])
        )
        ov_attention_mask = ov_outputs["image_attention_mask"]
        ov_num_img_tokens = ov_outputs["num_img_tokens"]

        pt_start = time.time()
        pt_output = pt_model(image)
        pt_time = time.time() - pt_start
        pt_embeds = pt_output["input_image_embeds"]
        pt_image_sizes = np.array((pt_output["image_height"], pt_output["image_width"]))
        pt_attention_mask = pt_output["image_attention_mask"]
        pt_num_img_tokens = pt_output["num_img_tokens"]

        original_start = time.time()
        original_outputs = original_model([to_pil_processor(image[0])])
        original_time = time.time() - original_start
        original_embeds = original_outputs["input_image_embeds"]
        original_image_sizes = original_outputs["image_sizes"]
        original_attention_mask = original_outputs["image_attention_mask"]
        original_num_img_tokens = original_outputs["num_img_tokens"]

        print(f"\n\n=============  image {i}  =============")
        print(
            f"ov time: {ov_time:.4f} pt time: {pt_time:.4f} original time: {original_time:.4f}"
        )

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


def get_position_ids_example_inputs(model):
    pairs = []
    for image in test_images:
        outputs = model([to_pil_processor(image[0])])
        embeds = outputs["input_image_embeds"]
        attention_mask = outputs["image_attention_mask"]
        pairs.append((embeds, attention_mask))
    return pairs


def get_position_ids_models(main_model):
    example_inputs = get_position_ids_example_inputs(main_model)
    pt_model = PositionIdsModel()
    pt_model(*example_inputs[0])

    scripted_model = torch.jit.script(pt_model, example_inputs=example_inputs)

    ov_model = ov.convert_model(
        scripted_model,
        example_input=example_inputs[0],
    )
    ov_model = ov.compile_model(ov_model, "CPU")

    return pt_model, ov_model


def test_position_ids_model(
    label: str,
    main_model,
    images: list[torch.Tensor],
):
    pt_model, ov_model = get_position_ids_models(main_model)

    print(f"====== {label} ====")

    for i, image in enumerate(images):
        main_model_outputs = main_model([to_pil_processor(image[0])])
        main_model_embeds = main_model_outputs["input_image_embeds"]
        main_model_attention_mask = main_model_outputs["image_attention_mask"]

        pt_start = time.time()
        pt_position_ids = pt_model(main_model_embeds, main_model_attention_mask)[
            "patch_position_ids"
        ]
        pt_time = time.time() - pt_start

        ov_start = time.time()
        ov_position_ids = ov_model(
            {
                "input_image_embeds.1": main_model_embeds,
                "image_attention_mask.1": main_model_attention_mask,
            }
        )["patch_position_ids"]
        ov_time = time.time() - ov_start

        print(f"\n\n=============  image {i}  =============")
        print(f"pt time: {pt_time:.4f} ov time: {ov_time:.4f}")
        print(
            f"patch_position_ids shapes: pt -> {pt_position_ids.shape}, ov -> {ov_position_ids.shape}"
        )
        assert np.all(np.array(pt_position_ids) == np.array(ov_position_ids))


test_images = get_test_images()

pil_images = [to_pil_processor(x[0]) for x in test_images]
original_model = OriginalModel()

# phi4_image_preprocessing_model = Phi4MMConvertableModel()
# phi4_image_preprocessing_model(test_images[0])


# scripted_model = torch.jit.script(
#     test_model, [pt_images, torch.tensor(utils.SORTED_TARGET_RATIOS)]
# )

# ov_model = ov.convert_model(
#     phi4_image_preprocessing_model,
#     example_input=get_test_images()[0],
# )

# ov_model_path = ".vscode/vlm/models/preprocess_image.xml"
# ov.save_model(ov_model, ov_model_path)

# ov_model = ov.compile_model(ov_model, "CPU")


# test_position_ids_model("position ids model pt vs ov", original_model, test_images)

# test_ov_model(
#     "covertable pytorch vs ov",
#     phi4_image_preprocessing_model,
#     original_model,
#     get_test_images(),
# )


def test_target_sizes_model():
    target_sizes_model = TargetSizesModel()
    target_sizes_model(test_images[1])

    scripted_model = torch.jit.script(target_sizes_model, [test_images[0]])

    ov_target_sizes_model = ov.convert_model(
        scripted_model, example_input=test_images[0]
    )
    ov_target_sizes_model = ov.compile_model(ov_target_sizes_model, "CPU")

    print(f"====== target sizes model ====")

    for i, image in enumerate(test_images):
        pt_start = time.perf_counter()
        pt_output = target_sizes_model(image)
        pt_time = time.perf_counter() - pt_start
        pt_new_size = pt_output["new_size"]
        pt_padding_width = pt_output["padding_width"]
        pt_padding_height = pt_output["padding_height"]
        pt_attention_mask = pt_output["attention_mask"]

        ov_start = time.perf_counter()
        ov_output = ov_target_sizes_model(image)
        ov_time = time.perf_counter() - ov_start
        ov_new_size = ov_output["new_size"]
        ov_padding_width = ov_output["padding_width"]
        ov_padding_height = ov_output["padding_height"]
        ov_attention_mask = ov_output["attention_mask"]

        print(f"\n\n=============  image {i}  =============")
        print(f"pt time: {pt_time:.4f} ov time: {ov_time:.4f}")
        print(f"new_size pt-> {pt_new_size}, ov -> {ov_new_size}")
        print(f"padding_width pt-> {pt_padding_width}, ov -> {ov_padding_width}")
        print(f"padding_height pt-> {pt_padding_height}, ov -> {ov_padding_height}")
        print(
            f"attention_mask shapes: pt -> {pt_attention_mask.shape}, ov -> {ov_attention_mask.shape}"
        )

        assert np.all(np.array(pt_new_size) == np.array(ov_new_size))
        assert np.all(np.array(pt_padding_width) == np.array(ov_padding_width))
        assert np.all(np.array(pt_padding_height) == np.array(ov_padding_height))
        assert np.all(np.array(pt_attention_mask) == np.array(ov_attention_mask))


# test_target_sizes_model()


def test_preprocess_model():
    target_sizes_model = TargetSizesModel()
    scripted_model = torch.jit.script(target_sizes_model, [test_images[0]])

    ov_target_sizes_model = ov.convert_model(
        scripted_model, example_input=test_images[0]
    )
    ov_target_sizes_model = ov.compile_model(ov_target_sizes_model, "CPU")

    ov_output = ov_target_sizes_model(test_images[0])
    ov_new_size = ov_output["new_size"]
    ov_padding_width = ov_output["padding_width"]
    ov_padding_height = ov_output["padding_height"]
    ov_attention_mask = ov_output["attention_mask"]

    example_input = (
        test_images[0],
        torch.tensor(ov_attention_mask),
        torch.tensor(ov_new_size),
        torch.tensor(ov_padding_width),
        torch.tensor(ov_padding_height),
    )

    preprocess_model = PreProcessModel()
    preprocess_model(*example_input)

    ov_preprocess_model = ov.convert_model(
        preprocess_model, example_input=example_input
    )
    ov_preprocess_model = ov.compile_model(ov_preprocess_model, "CPU")
    ov_preprocess_model(example_input)
