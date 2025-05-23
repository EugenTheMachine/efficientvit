import copy
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from segment_anything import SamPredictor#, SamAutomaticMaskGenerator
from segment_anything.modeling import Sam
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.utils.amg import build_all_layer_point_grids
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize, to_pil_image

from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone
from efficientvit.models.nn import (
    ConvLayer,
    DAGBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
    UpSampleLayer,
    build_norm,
)
from efficientvit.models.utils import build_kwargs_from_config, get_device

__all__ = [
    "SamPad",
    "SamResize",
    "SamNeck",
    "EfficientViTSamImageEncoder",
    "EfficientViTSam",
    "EfficientViTSamPredictor",
    "EfficientViTSamAutomaticMaskGenerator",
    "efficientvit_sam_l0",
    "efficientvit_sam_l1",
    "efficientvit_sam_l2",
    "efficientvit_sam_xl0",
    "efficientvit_sam_xl1",
]


class SamPad:
    def __init__(self, size: int, fill: float = 0, pad_mode="corner") -> None:
        self.size = size
        self.fill = fill
        self.pad_mode = pad_mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        th, tw = self.size, self.size
        assert th >= h and tw >= w
        if self.pad_mode == "corner":
            image = F.pad(image, (0, tw - w, 0, th - h), value=self.fill)
        else:
            raise NotImplementedError
        return image

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size},mode={self.pad_mode},fill={self.fill})"


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # print(image.shape)
        h, w = image.shape[-2:]
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        # print(f"Apply shape: {image.shape}")
        try:
            return np.array(resize(to_pil_image(image[0,:,:,:]), target_size))
        except IndexError:
            return np.array(resize(to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class SamNeck(DAGBlock):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        out_dim: int = 256,
        norm="bn2d",
        act_func="gelu",
    ):
        inputs = {}
        for fid, in_channel in zip(fid_list, in_channel_list):
            inputs[fid] = OpSequential(
                [
                    ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                    UpSampleLayer(size=(64, 64)),
                ]
            )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mb":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmb":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            elif middle_op == "res":
                block = ResBlock(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "sam_encoder": OpSequential(
                [
                    ConvLayer(
                        head_width,
                        out_dim,
                        1,
                        use_bias=True,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }

        super(SamNeck, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class EfficientViTSamImageEncoder(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone | EfficientViTLargeBackbone, neck: SamNeck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

        self.norm = build_norm("ln2d", 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        feed_dict = self.neck(feed_dict)

        output = feed_dict["sam_encoder"]
        output = self.norm(output)
        return output


class EfficientViTSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: EfficientViTSamImageEncoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                SamResize(self.image_size[1]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                    std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                ),
                SamPad(self.image_size[1]),
            ]
        )

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_size[0], self.image_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(
        self,
        batched_input: list[dict[str, Any]],
        multimask_output: bool,
    ):
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)

        image_embeddings = self.image_encoder(input_images)

        outputs = []
        iou_outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            outputs.append(low_res_masks)
            iou_outputs.append(iou_predictions)

        outputs = torch.stack([out for out in outputs], dim=0)
        iou_outputs = torch.stack(iou_outputs, dim=0)

        return outputs, iou_outputs


class EfficientViTSamPredictor:
    def __init__(self, sam_model: EfficientViTSam) -> None:
        self.model = sam_model
        self.reset_image()

    @property
    def transform(self):
        return self

    @property
    def device(self):
        return get_device(self.model)

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None

    def apply_coords(self, coords: np.ndarray, im_size=None) -> np.ndarray:
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, im_size=None) -> np.ndarray:
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4)

    def apply_boxes_torch(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4)

    def apply_coords_torch(self, coords: torch.Tensor, im_size=None) -> np.ndarray:
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        coords_copy = coords.detach().clone().to(torch.float)
        coords_copy[..., 0] = coords_copy[..., 0] * (new_w / old_w)
        coords_copy[..., 1] = coords_copy[..., 1] * (new_h / old_h)
        return coords_copy

    @torch.inference_mode()
    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        self.reset_image()

        self.original_size = image.shape[:2]
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.model.image_size[0]
        )

        torch_data = self.model.transform(image).unsqueeze(dim=0).to(get_device(self.model))
        # print(f"Shape: {torch_data.shape}")
        self.features = self.model.image_encoder(torch_data)
        self.is_image_set = True

    @torch.inference_mode()
    def set_image_batch(self, image: torch.Tensor) -> None:
        """
        image torch.Tensor : Shape (B,C,H,W) with data expected to be preprocessed already, see EfficientViTSam.transform for the expected transforms.
        """
        self.reset_image()

        original_height, original_width = image.shape[-2], image.shape[-1]
        self.original_size = (original_height, original_width)
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.model.image_size[0]
        )

        self.features = self.model.image_encoder(image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        device = get_device(self.model)
        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = self.apply_coords(point_coords)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.apply_boxes(box)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return masks, iou_predictions, low_res_masks

    @torch.inference_mode()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        image_index: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.
        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          image_index (int): If provided will be used to index the image embeddings used by the decoder. This needs to be used if one uses set_image_batch.
        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        if image_index is not None:
            image_embeddings = self.features[image_index].unsqueeze(0)
        else:
            image_embeddings = self.features
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.model.mask_threshold
        return masks, iou_predictions, low_res_masks


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        max_mask_region_area_ratio: float = 0.1,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.max_mask_region_area_ratio = max_mask_region_area_ratio
        self.output_mode = output_mode

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            area_ratio = ann["area"] / (image.shape[0] * image.shape[1])
            if area_ratio < self.max_mask_region_area_ratio:
                curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) and len(data["crop_boxes"]) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=False,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data


class EfficientViTSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    def __init__(
        self,
        model: EfficientViTSam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[list[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."

        self.predictor = EfficientViTSamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode


def build_efficientvit_sam(image_encoder: EfficientViTSamImageEncoder, image_size: int) -> EfficientViTSam:
    return EfficientViTSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        image_size=(1024, image_size),
    )


def efficientvit_sam_l0(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l0

    backbone = efficientvit_backbone_l0(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l1(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l2(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=12,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl0(image_size: int = 1024, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone

    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[0, 1, 1, 2, 3, 3],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )

    neck = SamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=6,
        expand_ratio=4,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl1(image_size: int = 1024, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone

    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 4, 6, 6],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )

    neck = SamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=12,
        expand_ratio=4,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)
