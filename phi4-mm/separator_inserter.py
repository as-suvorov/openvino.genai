import torch
import numpy as np
import openvino as ov
import math


def image_embed_to_trace(
    img_features, height, width, sub_GN, glb_GN
):
    # img_features = self.get_img_features(  # float32[5, 256, 1152]
    #     image_pixel_values.flatten(0, 1),
    #     image_attention_mask=image_attention_mask.flatten(0, 1).to(dtype=bool),
    # )
    # self.sub_GN float32[1, 1, 1, 1152]
    # self.glb_GN float32[1, 1, 1152]
    image_dim_out = img_features.shape[2]
    image_size = 448
    batch_size = 1

    base_feat_size = int(np.sqrt(img_features.shape[1]))
    img_features = img_features.view(batch_size, -1, base_feat_size**2, image_dim_out)

    output_imgs = []
    for idx in range(batch_size):
        height_ratio = height // image_size
        width_ratio = width // image_size
        area_ratio = height_ratio * width_ratio

        global_img = img_features[idx, :1]
        global_img = global_img.reshape(1, base_feat_size, base_feat_size, image_dim_out).contiguous()
        temporary_extensor = sub_GN.repeat(1, base_feat_size, 1, 1)
        global_img = torch.cat([global_img, temporary_extensor], dim=2).reshape(1, -1, image_dim_out)

        sub_img = img_features[idx, 1:]
        sub_img = sub_img[:area_ratio]
        sub_img = (
            sub_img.reshape(height_ratio, width_ratio, base_feat_size, base_feat_size, image_dim_out)
            .transpose(1, 2)
            .reshape(1, height_ratio * base_feat_size, width_ratio * base_feat_size, image_dim_out)
            .contiguous()
        )

        temporary_extensor = sub_GN.repeat(1, height_ratio * base_feat_size, 1, 1)

        sub_img = torch.cat([sub_img, temporary_extensor], dim=2).reshape(1, -1, image_dim_out)

        # Merge global and sub
        output_imgs.append(torch.cat([sub_img, glb_GN, global_img], dim=1))
    return output_imgs

class SeparatorInserter(torch.nn.Module):
    def forward(self, img_features, height, width, sub_GN, glb_GN):
        img_features = img_features[0]
        image_dim_out = img_features.shape[2]
        image_size = 448
        batch_size = 1

        # base_feat_size = int(np.sqrt(img_features.shape[1]))
        base_feat_size = math.floor(math.sqrt(img_features.shape[1]))
        img_features = img_features.view(batch_size, -1, base_feat_size**2, image_dim_out)

        idx = 0
        height_ratio = height // image_size
        width_ratio = width // image_size
        area_ratio = height_ratio * width_ratio

        global_img = img_features[idx, :1]
        global_img = global_img.reshape(1, base_feat_size, base_feat_size, image_dim_out).contiguous()
        temporary_extensor = sub_GN.repeat(1, base_feat_size, 1, 1)
        global_img = torch.cat([global_img, temporary_extensor], dim=2).reshape(1, -1, image_dim_out)

        sub_img = img_features[idx, 1:]
        sub_img = sub_img[:area_ratio]
        sub_img = (
            sub_img.reshape(height_ratio, width_ratio, base_feat_size, base_feat_size, image_dim_out)
            .transpose(1, 2)
            .reshape(1, height_ratio * base_feat_size, width_ratio * base_feat_size, image_dim_out)
            .contiguous()
        )

        temporary_extensor = sub_GN.repeat(1, height_ratio * base_feat_size, 1, 1)

        sub_img = torch.cat([sub_img, temporary_extensor], dim=2).reshape(1, -1, image_dim_out)

        # Merge global and sub
        output_img = torch.cat([sub_img, glb_GN, global_img], dim=1)
        return output_img

def test(image_embed_to_trace, inserter, test_tiny_img_features, test_tiny_height, test_tiny_width, test_tiny_sub_GN, test_tiny_glb_G):
    tiny_reference = image_embed_to_trace(test_tiny_img_features, test_tiny_height, test_tiny_width, test_tiny_sub_GN, test_tiny_glb_GN)
    tiny_ov_prediction = inserter.infer_new_request((test_tiny_img_features[None], test_tiny_height, test_tiny_width, test_tiny_sub_GN, test_tiny_glb_GN))
    tiny_ov_prediction = next(iter(tiny_ov_prediction.values()))
    assert tiny_ov_prediction.shape == tiny_reference[0].shape
    assert (tiny_ov_prediction == tiny_reference[0]).all()


pt_model = SeparatorInserter()
test_img_features = torch.randn(5, 256, 1152, dtype=torch.float32)
height = torch.tensor(896, dtype=torch.int32)
width = torch.tensor(896, dtype=torch.int32)
test_sub_GN = torch.randn(1, 1, 1, 1152, dtype=torch.float32)
test_glb_GN = torch.randn(1, 1, 1152, dtype=torch.float32)
ov_model = ov.convert_model(
    pt_model,
    example_input={
        'img_features': test_img_features[None],
        'height': height,
        'width': width,
        'sub_GN': test_sub_GN,
        'glb_GN': test_glb_GN
    },
    input={
        "img_features": ov.PartialShape([1, -1, -1, -1]),
        "height": ov.PartialShape([]),
        "width": ov.PartialShape([]),
        "sub_GN": ov.PartialShape([1, 1, 1, -1]),
        "glb_GN": ov.PartialShape([1, 1, -1]),
    }
)
print(ov_model)
ov.save_model(ov_model, "separator_ref/a.xml", compress_to_fp16=False)
reference = image_embed_to_trace(test_img_features, height, width, test_sub_GN, test_glb_GN)

pt_prediction = pt_model(test_img_features[None], height, width, test_sub_GN, test_glb_GN)
assert (pt_prediction[0] == reference[0]).all()
inserter = ov.Core().compile_model(ov_model, "CPU")
inserter = ov.Core().compile_model('separator_res/a.xml', "CPU")
ov_prediction = inserter.infer_new_request((test_img_features[None], height, width, test_sub_GN, test_glb_GN))
ov_prediction = next(iter(ov_prediction.values()))
assert ov_prediction.shape == reference[0].shape
assert (ov_prediction == reference[0]).all()

# tiny-random-phi-4-multimodal with 14x14 input image
test_tiny_img_features = torch.randn(2, 256, 16, dtype=torch.float32)
test_tiny_height = torch.tensor(448, dtype=torch.int32)
test_tiny_width = torch.tensor(448, dtype=torch.int32)
test_tiny_sub_GN = torch.randn(1, 1, 1, 16, dtype=torch.float32)
test_tiny_glb_GN = torch.randn(1, 1, 16, dtype=torch.float32)
test(image_embed_to_trace, inserter, test_tiny_img_features, test_tiny_height, test_tiny_width, test_tiny_sub_GN, test_tiny_glb_GN)
test_tiny_img_features = torch.randn(2, 256, 12, dtype=torch.float32)
test_tiny_height = torch.tensor(448, dtype=torch.int32)
test_tiny_width = torch.tensor(448, dtype=torch.int32)
test_tiny_sub_GN = torch.randn(1, 1, 1, 12, dtype=torch.float32)
test_tiny_glb_GN = torch.randn(1, 1, 12, dtype=torch.float32)
test(image_embed_to_trace, inserter, test_tiny_img_features, test_tiny_height, test_tiny_width, test_tiny_sub_GN, test_tiny_glb_GN)
# base_feat_size = math.floor(math.sqrt(img_features.shape[1])) isn't traced and saved as const. An extra input can be used to workaround if we need it.
# test_tiny_img_features = torch.randn(2, 121, 12, dtype=torch.float32)
# test_tiny_height = torch.tensor(448, dtype=torch.int32)
# test_tiny_width = torch.tensor(448, dtype=torch.int32)
# test_tiny_sub_GN = torch.randn(1, 1, 1, 12, dtype=torch.float32)
# test_tiny_glb_GN = torch.randn(1, 1, 12, dtype=torch.float32)
# test(image_embed_to_trace, inserter, test_tiny_img_features, test_tiny_height, test_tiny_width, test_tiny_sub_GN, test_tiny_glb_GN)
