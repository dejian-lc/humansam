from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)

from .network.encoder import DepthProEncoder
from .network.vit_factory import VIT_CONFIG_DICT, ViTPreset, create_vit


@dataclass
class DepthProEncoderConfig:
    """Configuration for DepthProEncoder."""

    patch_encoder_preset: ViTPreset
    image_encoder_preset: ViTPreset
    decoder_features: int

    checkpoint_uri: Optional[str] = None
    fov_encoder_preset: Optional[ViTPreset] = None
    use_fov_head: bool = True


DEFAULT_ENCODER_CONFIG_DICT = DepthProEncoderConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    checkpoint_uri="/home/lc/workspace/InternVideo/InternVideo2/single_modality/models/checkpoints/depth_pro.pt",
    decoder_features=256,
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384",
)


def create_backbone_model(
    preset: ViTPreset
) -> Tuple[nn.Module, ViTPreset]:
    """Create and load a backbone model given a config.

    Args:
    ----
        preset: A backbone preset to load pre-defind configs.

    Returns:
    -------
        A Torch module and the associated config.

    """
    if preset in VIT_CONFIG_DICT:
        config = VIT_CONFIG_DICT[preset]
        model = create_vit(preset=preset, use_pretrained=False)
    else:
        raise KeyError(f"Preset {preset} not found.")

    return model, config


def create_encoder_model_and_transforms(
    config: DepthProEncoderConfig = DEFAULT_ENCODER_CONFIG_DICT,
    device: torch.device = torch.device("cpu"),
    precision: torch.dtype = torch.half,
    checkpoint_path: Optional[str] = None,
) -> Tuple[DepthProEncoder, Compose]:
    """Create a DepthProEncoder model and load weights from `config.checkpoint_uri`.

    Args:
    ----
        config: The configuration for the DepthProEncoder architecture.
        device: The optional Torch device to load the model onto, default runs on "cpu".
        precision: The optional precision used for the model, default is FP32.
        checkpoint_path: Optional path to the checkpoint file. If provided, overrides config.checkpoint_uri.

    Returns:
    -------
        The Torch DepthProEncoder model and associated Transform.

    """
    patch_encoder, patch_encoder_config = create_backbone_model(
        preset=config.patch_encoder_preset
    )
    image_encoder, _ = create_backbone_model(
        preset=config.image_encoder_preset
    )

    dims_encoder = patch_encoder_config.encoder_feature_dims
    hook_block_ids = patch_encoder_config.encoder_feature_layer_ids
    encoder = DepthProEncoder(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=hook_block_ids,
        decoder_features=config.decoder_features,
    ).to(device)

    if precision == torch.half:
        encoder.half()

    # torch.float32

    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: x.to(device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConvertImageDtype(precision),
        ]
    )

    checkpoint_uri = checkpoint_path if checkpoint_path is not None else config.checkpoint_uri

    if checkpoint_uri is not None:
        state_dict = torch.load(checkpoint_uri, map_location="cpu")

        # Create a mapping function to map keys in the weight file to keys in the model structure
        def map_state_dict_keys(state_dict_o):
            new_state_dict = {}
            for key, value in state_dict_o.items():
                if key.startswith("encoder."):
                    new_key = key[len("encoder."):]
                    new_state_dict[new_key] = value
            return new_state_dict

        state_dict = map_state_dict_keys(state_dict)

        encoder.load_state_dict(state_dict, strict=False)

    return encoder, transform


class MinimalDepthProEncoder(nn.Module):
    """Minimal DepthProEncoder network that returns the smallest feature map."""

    def __init__(
        self,
        encoder: DepthProEncoder,
    ):
        """Initialize MinimalDepthProEncoder.

        Args:
        ----
            encoder: The DepthProEncoder backbone.

        """
        super().__init__()
        self.encoder = encoder

    @property
    def img_size(self) -> int:
        """Return the internal image size of the network."""
        return self.encoder.img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input image and return the smallest feature map.

        Args:
        ----
            x (torch.Tensor): Input image.

        Returns:
        -------
            The smallest feature map from the encoder.

        """
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        encodings = self.encoder(x)
        # Return the smallest feature map
        return encodings[-1]

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        interpolation_mode="bilinear",
    ) -> torch.Tensor:
        """Infer the smallest feature map for a given image.

        If the image is not at network resolution, it is resized to the network resolution.

        Args:
        ----
            x (torch.Tensor): Input image
            interpolation_mode (str): Interpolation function for downsampling/upsampling.

        Returns:
        -------
            The smallest feature map.

        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        _, _, H, W = x.shape
        resize = H != self.img_size or W != self.img_size

        if resize:
            x = nn.functional.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode=interpolation_mode,
                align_corners=False,
            )

        smallest_feature_map = self.forward(x)

        return smallest_feature_map