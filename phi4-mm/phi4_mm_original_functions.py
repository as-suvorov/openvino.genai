import math
import torch
import torchvision
import torchvision.transforms.functional


# inputs: images, text

# text_template: text -> <user><image_1>...text...<assistant>

# tokenizer: text_template -> tokens: <user_token><image_1_token_placeholder>...tokens...<assistant_token><eos>

# text embeddings: text_template -> text_embeddings

# ==========================
# image_preprocessing: image -> image_preprocessor.preprocess(image) -> processed_image_tensor {input_image_embeds[1, 7, 3, 448, 448], ...}
# ==========================

# vision_encoding: processed_image_tensor -> vision_encoder -> image_embeddings

# vision projector model: image_embeddings -> image_embeddings_projection

# merge text_embeddings with image_embeddings_projection -> image_embeddings + text_embeddings -> inputs_embeds

# text generation: inputs_embeds -> llm -> tokens output


def pad_mask_to_max_num_crops(masks, max_crops=5):
    B, H, W = masks.shape
    if B < max_crops:
        pad = torch.ones(max_crops - B, H, W, dtype=masks.dtype, device=masks.device)
        masks = torch.cat([masks, pad], dim=0)
    return masks


def pad_to_max_num_crops(images, max_crops=5):
    """
    images: B x 3 x H x W, B<=max_crops
    """
    B, _, H, W = images.shape
    if B < max_crops:
        pad = torch.zeros(
            max_crops - B, 3, H, W, dtype=images.dtype, device=images.device
        )
        images = torch.cat([images, pad], dim=0)
    return images


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image,
    min_num=1,
    max_num=12,
    image_size=384,
    mask_size=27,
):
    orig_width, orig_height = image.size

    w_crop_num = math.ceil(orig_width / float(image_size))
    h_crop_num = math.ceil(orig_height / float(image_size))

    if w_crop_num * h_crop_num > max_num:

        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
    else:
        target_width = image_size * w_crop_num
        target_height = image_size * h_crop_num
        target_aspect_ratio = (w_crop_num, h_crop_num)

    # Calculate the ratio
    ratio_width = target_width / orig_width
    ratio_height = target_height / orig_height
    if ratio_width < ratio_height:
        new_size = (target_width, int(orig_height * ratio_width))
        padding_width = 0
        padding_height = target_height - int(orig_height * ratio_width)
    else:
        new_size = (int(orig_width * ratio_height), target_height)
        padding_width = target_width - int(orig_width * ratio_height)
        padding_height = 0

    attention_mask = torch.ones(
        (
            int(mask_size * target_aspect_ratio[1]),
            int(mask_size * target_aspect_ratio[0]),
        )
    )
    if padding_width >= 14:
        attention_mask[:, -math.floor(padding_width / 14) :] = 0
    if padding_height >= 14:
        attention_mask[-math.floor(padding_height / 14) :, :] = 0
    assert attention_mask.sum() > 0

    if min(new_size[1], target_height) < 10 or min(new_size[0], target_width) < 10:
        raise ValueError(f"the aspect ratio is very extreme {new_size}")

    image = torchvision.transforms.functional.resize(
        image,
        [new_size[1], new_size[0]],
    )

    resized_img = torchvision.transforms.functional.pad(
        image, [0, 0, padding_width, padding_height], fill=[255, 255, 255]
    )

    return resized_img, attention_mask
