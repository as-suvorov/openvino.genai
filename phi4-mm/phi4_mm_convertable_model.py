import torch
from torchvision.transforms import v2
import torchvision.transforms.v2
import torchvision


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


class TargetSizesModel(torch.nn.Module):
    def find_closest_aspect_ratio(
        self,
        width: torch.Tensor,
        height: torch.Tensor,
        image_size: torch.Tensor,
    ):
        aspect_ratio = width / height
        target_ratios = torch.tensor(
            [
                (1, 1),
                (1, 2),
                (2, 1),
                (3, 1),
                (1, 3),
                (2, 2),
                (4, 1),
                (1, 4),
                (5, 1),
                (1, 5),
                (6, 1),
                (3, 2),
                (2, 3),
                (1, 6),
                (7, 1),
                (1, 7),
                (4, 2),
                (2, 4),
                (1, 8),
                (8, 1),
                (9, 1),
                (3, 3),
                (1, 9),
                (1, 10),
                (5, 2),
                (10, 1),
                (2, 5),
                (1, 11),
                (11, 1),
                (6, 2),
                (1, 12),
                (2, 6),
                (12, 1),
                (3, 4),
                (4, 3),
                (13, 1),
                (1, 13),
                (14, 1),
                (1, 14),
                (7, 2),
                (2, 7),
                (1, 15),
                (5, 3),
                (3, 5),
                (15, 1),
                (8, 2),
                (16, 1),
                (4, 4),
                (1, 16),
                (2, 8),
                (1, 17),
                (17, 1),
                (18, 1),
                (3, 6),
                (9, 2),
                (1, 18),
                (6, 3),
                (2, 9),
                (1, 19),
                (19, 1),
                (20, 1),
                (5, 4),
                (2, 10),
                (1, 20),
                (4, 5),
                (10, 2),
                (7, 3),
                (1, 21),
                (3, 7),
                (21, 1),
                (2, 11),
                (11, 2),
                (22, 1),
                (1, 22),
                (1, 23),
                (23, 1),
                (1, 24),
                (24, 1),
                (6, 4),
                (3, 8),
                (4, 6),
                (2, 12),
                (8, 3),
                (12, 2),
                (5, 5),
                (1, 25),
                (25, 1),
                (1, 26),
                (2, 13),
                (13, 2),
                (26, 1),
                (9, 3),
                (27, 1),
                (3, 9),
                (1, 27),
                (1, 28),
                (7, 4),
                (14, 2),
                (2, 14),
                (4, 7),
                (28, 1),
                (29, 1),
                (1, 29),
                (3, 10),
                (15, 2),
                (1, 30),
                (2, 15),
                (10, 3),
                (5, 6),
                (30, 1),
                (6, 5),
                (1, 31),
                (31, 1),
                (8, 4),
                (1, 32),
                (4, 8),
                (32, 1),
                (16, 2),
                (2, 16),
                (1, 33),
                (33, 1),
                (3, 11),
                (11, 3),
                (34, 1),
                (17, 2),
                (2, 17),
                (1, 34),
                (35, 1),
                (1, 35),
                (7, 5),
                (5, 7),
                (4, 9),
                (6, 6),
                (3, 12),
                (18, 2),
                (12, 3),
                (36, 1),
                (1, 36),
                (9, 4),
                (2, 18),
            ],
            dtype=torch.int64,
        )

        best_ratio_diff = torch.tensor(torch.inf)
        best_ratio = torch.tensor([1, 1], dtype=torch.int64)
        area = width * height

        target_aspect_ratio = aspect_ratio

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1].to(torch.float32)
            ratio_diff = torch.abs(aspect_ratio - target_aspect_ratio).to(torch.float32)
            if torch.lt(ratio_diff, best_ratio_diff):
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif torch.eq(ratio_diff, best_ratio_diff):
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def forward(self, image: torch.Tensor):
        # nhwc format
        dynamic_hd = torch.tensor(36).int()
        dyhd_base_resolution = torch.tensor(448).int()

        mask_resolution = dyhd_base_resolution // 14

        _, orig_height, orig_width, _ = image.shape
        orig_height = torch.tensor(orig_height, dtype=torch.int64)
        orig_width = torch.tensor(orig_width, dtype=torch.int64)

        w_crop_num = torch.ceil(orig_width / dyhd_base_resolution.to(torch.float)).to(
            torch.int64
        )

        h_crop_num = torch.ceil(orig_height / dyhd_base_resolution.to(torch.float)).to(
            torch.int64
        )

        target_aspect_ratio = self.find_closest_aspect_ratio(
            orig_width,
            orig_height,
            dyhd_base_resolution,
        )

        if torch.gt(w_crop_num * h_crop_num, dynamic_hd):
            # find the closest aspect ratio to the target
            target_width = dyhd_base_resolution * target_aspect_ratio[0]
            target_height = dyhd_base_resolution * target_aspect_ratio[1]
        else:
            target_width = dyhd_base_resolution * w_crop_num
            target_height = dyhd_base_resolution * h_crop_num
            target_aspect_ratio = torch.tensor([w_crop_num, h_crop_num])

        ratio_width = target_width / orig_width
        ratio_height = target_height / orig_height

        if ratio_width < ratio_height:
            new_size = torch.tensor(
                [target_width, (orig_height * ratio_width).to(torch.int64)]
            )
            padding_width = 0
            padding_height = int(target_height - int(orig_height * ratio_width))
        else:
            new_size = torch.tensor(
                [(orig_width * ratio_height).to(torch.int64), target_height]
            )
            padding_width = int(target_width - int(orig_width * ratio_height))
            padding_height = 0

        attention_mask = torch.ones(
            (
                int(mask_resolution * target_aspect_ratio[1]),
                int(mask_resolution * target_aspect_ratio[0]),
            )
        )

        if padding_width >= 14:
            attention_mask[:, -(padding_width // 14) :] = 0
        if padding_height >= 14:
            attention_mask[-(padding_height // 14) :, :] = 0

        return {
            "new_size": new_size,
            "padding_width": torch.tensor(padding_width),
            "padding_height": torch.tensor(padding_height),
            "attention_mask": attention_mask,
        }


class PreProcessModel(torch.nn.Module):
    def forward(
        self,
        image: torch.Tensor,
        attention_mask: torch.Tensor,
        new_size: torch.Tensor,
        padding_width: torch.Tensor,
        padding_height: torch.Tensor,
    ):
        # nhwc format
        image = image[0].to(torch.float32) / 255.0
        # hwc -> chw
        image = image.permute(2, 0, 1).contiguous()

        dyhd_base_resolution = torch.tensor(448).int()

        base_resolution = dyhd_base_resolution
        mask_resolution = base_resolution // 14
        new_height_size = new_size[1]
        new_width_size = new_size[0]
        resize = torch.nn.Sequential(
            torchvision.transforms.v2.Resize([new_height_size, new_width_size]),  # type: ignore
        )

        hd_image = resize(image)

        # pad
        image_pad_width_tensor = torch.ones([3, new_height_size, padding_width])  # type: ignore
        image_pad_height_tensor = torch.ones([3, padding_height, padding_width + new_width_size])  # type: ignore

        hd_image = torch.concat([hd_image, image_pad_width_tensor], dim=2)
        hd_image = torch.concat([hd_image, image_pad_height_tensor], dim=1)

        hd_image = v2.functional.normalize(hd_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

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
            + 16,
            dtype=torch.int64,
        )

        hd_image_reshape = torch.cat([global_image] + [hd_image_reshape], dim=0)
        hd_masks_reshape = torch.cat(
            [global_attention_mask] + [attention_mask_reshape], dim=0
        ).to(torch.bool)

        # pad_to_max_num_crops, pad_mask_to_max_num_crops skipped as it's a batch feature

        return {
            "input_image_embeds": hd_image_reshape.unsqueeze(0),
            "image_height": torch.tensor(IMAGE_H, dtype=torch.int64),
            "image_width": torch.tensor(IMAGE_W, dtype=torch.int64),
            "image_attention_mask": hd_masks_reshape.unsqueeze(0),
            "num_img_tokens": num_img_tokens.unsqueeze(0),
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

        hd_image = v2.functional.normalize(hd_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

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
