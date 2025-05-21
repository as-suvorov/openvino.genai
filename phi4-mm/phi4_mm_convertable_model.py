import torch
from torchvision.transforms import v2
import torchvision.transforms.v2


class PositionIdsModel(torch.nn.Module):
    def forward(
        self, input_image_embeds: torch.Tensor, image_attention_mask: torch.Tensor
    ):
        patch_size = 14
        num_patches_per_side = 32
        input_image_embeds = input_image_embeds.flatten(0, 1)
        image_attention_mask = image_attention_mask.flatten(0, 1)
        batch_size = input_image_embeds.shape[0]
        max_im_h, max_im_w = input_image_embeds.size(2), input_image_embeds.size(3)
        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // patch_size,
            max_im_w // patch_size,
        )
        boundaries = torch.arange(
            1 / num_patches_per_side, 1.0, 1 / num_patches_per_side
        )
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(image_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(
                fractional_coords_h, boundaries, right=True
            )
            bucket_coords_w = torch.bucketize(
                fractional_coords_w, boundaries, right=True
            )
            pos_ids = (
                bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w
            ).flatten()

            position_ids[batch_idx][p_attn_mask.view(-1).to(torch.bool)] = pos_ids

        return {
            "patch_position_ids": position_ids,
        }


class Phi4MMConvertableModel(torch.nn.Module):

    def dynamic_preprocess(
        self,
        image: torch.Tensor,
        min_num=torch.tensor(1).int(),
        max_num=torch.tensor(36).int(),
        image_size=torch.tensor(448).int(),
        mask_size=torch.tensor(32).int(),
    ):
        _, orig_height, orig_width = image.shape
        orig_height = torch.tensor(orig_height, dtype=torch.int64)
        orig_width = torch.tensor(orig_width, dtype=torch.int64)

        w_crop_num = torch.ceil(orig_width / image_size.to(torch.float))

        h_crop_num = torch.ceil(orig_height / image_size.to(torch.float))

        image = v2.functional.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # todo:
        # w_crop_num * h_crop_num > max_num condition is skipped

        target_width = image_size * w_crop_num
        target_height = image_size * h_crop_num
        target_aspect_ratio = (w_crop_num, h_crop_num)

        ratio_width = target_width / orig_width
        ratio_height = target_height / orig_height

        height_is_bigger = (ratio_width < ratio_height).to(torch.int64)
        height_is_not_bigger = (ratio_width >= ratio_height).to(torch.int64)

        new_width_size = height_is_bigger * target_width.to(torch.int64) + (
            height_is_not_bigger
        ) * (orig_width * ratio_height).to(torch.int64)
        new_height_size = height_is_bigger * (orig_height * ratio_width).to(
            torch.int64
        ) + (height_is_not_bigger) * target_height.to(torch.int64)

        padding_width = (
            height_is_bigger * 0
            + height_is_not_bigger
            * (target_width - (orig_width * ratio_height).int()).int()
        ).int()

        padding_height = (
            height_is_bigger * (target_height - (orig_height * ratio_width).int()).int()
            + height_is_not_bigger * 0
        ).int()

        mask_height = torch.tensor(mask_size * target_aspect_ratio[1]).int()
        mask_width = torch.tensor(mask_size * target_aspect_ratio[0]).int()
        attention_mask = torch.ones([mask_height, mask_width])  # type: ignore

        attention_mask[:, mask_width - (padding_width / 14).int() :] = 0
        attention_mask[mask_height - (padding_height / 14).int() :, :] = 0

        resize = torch.nn.Sequential(
            torchvision.transforms.v2.Resize([new_height_size, new_width_size]),  # type: ignore
        )

        image = resize(image)

        # pad
        image_pad_width_tensor = torch.ones([3, new_height_size, padding_width])  # type: ignore
        image_pad_height_tensor = torch.ones([3, padding_height, padding_width + new_width_size])  # type: ignore

        image = torch.concat([image, image_pad_width_tensor], dim=2)
        image = torch.concat([image, image_pad_height_tensor], dim=1)

        return image, attention_mask

    def forward(self, image: torch.Tensor):
        image = image[0].to(torch.float32) / 255.0
        # hwc -> chw
        image = image.permute(2, 0, 1).contiguous()

        dynamic_hd = torch.tensor(36).int()
        dyhd_base_resolution = torch.tensor(448).int()

        base_resolution = dyhd_base_resolution
        mask_resolution = base_resolution // 14

        hd_image, attention_mask = self.dynamic_preprocess(
            image,
            max_num=dynamic_hd,
            image_size=dyhd_base_resolution,
            mask_size=mask_resolution,
        )

        global_image = torch.nn.functional.interpolate(
            hd_image.unsqueeze(0).float(),
            size=(base_resolution, base_resolution),
            mode="bicubic",
        ).to(hd_image.dtype)

        _, IMAGE_H, IMAGE_W = hd_image.shape

        mask_shape = attention_mask.shape

        global_attention_mask = torch.ones((1, mask_resolution, mask_resolution))  # type: ignore
        hd_image_reshape = (
            hd_image.reshape(
                1,
                3,
                IMAGE_H // base_resolution,
                base_resolution,
                IMAGE_W // base_resolution,
                base_resolution,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, 3, base_resolution, base_resolution)
            .contiguous()
        )

        attention_mask_reshape = (
            attention_mask.reshape(
                1,
                mask_shape[0] // mask_resolution,
                mask_resolution,
                mask_shape[1] // mask_resolution,
                mask_resolution,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, mask_resolution, mask_resolution)
            .contiguous()
        )

        downsample_attention_mask = (
            attention_mask_reshape[:, 0::2, 0::2]
            .reshape(
                1,
                mask_shape[0] // mask_resolution,
                mask_shape[1] // mask_resolution,
                mask_resolution // 2 + mask_resolution % 2,
                mask_resolution // 2 + mask_resolution % 2,
            )
            .permute(0, 1, 3, 2, 4)
        )

        dam_shape = downsample_attention_mask.shape
        downsample_attention_mask = downsample_attention_mask.reshape(
            dam_shape[1] * dam_shape[2], dam_shape[3] * dam_shape[4]
        )

        num_img_tokens = torch.tensor(
            256
            + 1
            + downsample_attention_mask.sum()
            + downsample_attention_mask[:, 0].sum()
            + 16
        )

        hd_image_reshape = torch.cat([global_image] + [hd_image_reshape], dim=0)
        hd_masks_reshape = torch.cat(
            [global_attention_mask] + [attention_mask_reshape], dim=0
        )

        # pad_to_max_num_crops, pad_mask_to_max_num_crops skipped as it's a batch feature

        return {
            "input_image_embeds": hd_image_reshape.unsqueeze(0),
            "image_height": torch.tensor(IMAGE_H),
            "image_width": torch.tensor(IMAGE_W),
            "image_attention_mask": hd_masks_reshape.unsqueeze(0),
            "num_img_tokens": num_img_tokens.unsqueeze(0),
        }
