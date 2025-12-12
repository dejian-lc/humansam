from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

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
    checkpoint_uri="checkpoints/depth_model/depth_pro.pt",
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


class DepthProEncoderWrapper(nn.Module):
    """Wrapper for DepthProEncoder that returns the smallest feature map."""

    def __init__(self, encoder: DepthProEncoder):
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
        encodings = self.encoder(x)
        # Return the smallest feature map
        return encodings[-1]


def create_depth_pro_encoder(
    config: DepthProEncoderConfig = DEFAULT_ENCODER_CONFIG_DICT,
    device: torch.device = torch.device("cpu"),
    precision: torch.dtype = torch.half,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """Create a DepthProEncoder model and load weights from `config.checkpoint_uri`.

    Args:
    ----
        config: The configuration for the DepthProEncoder architecture.
        device: The optional Torch device to load the model onto, default runs on "cpu".
        precision: The optional precision used for the model, default is FP32.
        checkpoint_path: Optional path to the checkpoint file. If provided, overrides config.checkpoint_uri.

    Returns:
    -------
        The Torch DepthProEncoder model (wrapped).

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

    checkpoint_uri = checkpoint_path if checkpoint_path is not None else config.checkpoint_uri

    if checkpoint_uri is not None:
        state_dict = torch.load(checkpoint_uri, map_location="cpu")

        # Create a mapping function to map key names in the weight file to key names in the model structure
        def map_state_dict_keys(state_dict_o):
            new_state_dict = {}
            for key, value in state_dict_o.items():
                if key.startswith("encoder."):
                    new_key = key[len("encoder."):]
                    new_state_dict[new_key] = value
            return new_state_dict

        state_dict = map_state_dict_keys(state_dict)

        encoder.load_state_dict(state_dict, strict=False)

    return DepthProEncoderWrapper(encoder)
