import torch
import phi4_mm_original_functions
from PIL import Image
import torchvision


class OriginalModel(torch.nn.Module):
    def forward(
        self,
        images: list[Image.Image],
        dynamic_hd=36,
    ):
        """
        Args:
            images (`list["PIL.Image.Image"]`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        """

        # Basic settings.
        img_processor = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dyhd_base_resolution = 448

        # Dynamic HD
        base_resolution = dyhd_base_resolution
        images = [image.convert("RGB") for image in images]
        # cover 384 and 448 resolution
        mask_resolution = base_resolution // 14
        elems, image_attention_masks = [], []
        for im in images:
            elem, attention_mask = phi4_mm_original_functions.dynamic_preprocess(
                im,
                max_num=dynamic_hd,
                image_size=base_resolution,
                mask_size=mask_resolution,
            )
            # elem.save("original.jpg")
            elems.append(elem)
            image_attention_masks.append(attention_mask)
        hd_images = [img_processor(im) for im in elems]

        global_image = [
            torch.nn.functional.interpolate(
                im.unsqueeze(0).float(),
                size=(base_resolution, base_resolution),
                mode="bicubic",
            ).to(im.dtype)
            for im in hd_images
        ]

        shapes = [[im.size(1), im.size(2)] for im in hd_images]
        mask_shapes = [[mask.size(0), mask.size(1)] for mask in image_attention_masks]
        global_attention_mask = [
            torch.ones((1, mask_resolution, mask_resolution)) for _ in hd_images
        ]
        hd_images_reshape = [
            im.reshape(
                1,
                3,
                h // base_resolution,
                base_resolution,
                w // base_resolution,
                base_resolution,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, 3, base_resolution, base_resolution)
            .contiguous()
            for im, (h, w) in zip(hd_images, shapes)
        ]

        attention_masks_reshape = [
            mask.reshape(
                1,
                h // mask_resolution,
                mask_resolution,
                w // mask_resolution,
                mask_resolution,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, mask_resolution, mask_resolution)
            .contiguous()
            for mask, (h, w) in zip(image_attention_masks, mask_shapes)
        ]

        downsample_attention_masks = [
            mask[:, 0::2, 0::2]
            .reshape(
                1,
                h // mask_resolution,
                w // mask_resolution,
                mask_resolution // 2 + mask_resolution % 2,
                mask_resolution // 2 + mask_resolution % 2,
            )
            .permute(0, 1, 3, 2, 4)
            for mask, (h, w) in zip(attention_masks_reshape, mask_shapes)
        ]

        downsample_attention_masks = [
            mask.reshape(mask.size(1) * mask.size(2), mask.size(3) * mask.size(4))
            for mask in downsample_attention_masks
        ]

        num_img_tokens = [
            256 + 1 + int(mask.sum().item()) + int(mask[:, 0].sum().item()) + 16
            for mask in downsample_attention_masks
        ]

        hd_images_reshape = [
            torch.cat([_global_image] + [_im], dim=0)
            for _global_image, _im in zip(global_image, hd_images_reshape)
        ]
        hd_masks_reshape = [
            torch.cat([_global_mask] + [_mask], dim=0)
            for _global_mask, _mask in zip(
                global_attention_mask, attention_masks_reshape
            )
        ]

        max_crops = max([img.size(0) for img in hd_images_reshape])
        image_transformed = [
            phi4_mm_original_functions.pad_to_max_num_crops(im, max_crops)
            for im in hd_images_reshape
        ]

        image_transformed = torch.stack(image_transformed, dim=0)
        mask_transformed = [
            phi4_mm_original_functions.pad_mask_to_max_num_crops(mask, max_crops)
            for mask in hd_masks_reshape
        ]
        mask_transformed = torch.stack(mask_transformed, dim=0)

        returned_input_image_embeds = image_transformed
        returned_image_sizes = torch.tensor(shapes, dtype=torch.long)
        returned_image_attention_mask = mask_transformed
        returned_num_img_tokens = num_img_tokens

        data = {
            "input_image_embeds": returned_input_image_embeds,
            "image_sizes": returned_image_sizes[0],
            "image_attention_mask": returned_image_attention_mask,
            "num_img_tokens": torch.tensor(returned_num_img_tokens),
        }

        return data
