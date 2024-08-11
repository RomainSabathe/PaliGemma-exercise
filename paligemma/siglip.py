from dataclasses import dataclass

import einops
import torch
import torch.nn as nn


@dataclass
class ViTConfiguration:
    image_size: int = 224
    in_channels: int = 3
    hidden_size: int = 1_024  # "latent vector size D" in the paper
    patch_size: int = 16

    @property
    def num_patches(self) -> int:
        return self.image_size**2 // self.patch_size**2


@dataclass
class SigLipVisionConfiguration:
    encoder_configuration: ViTConfiguration


class SigLipVisionModel(nn.Module):
    def __init__(self, configuration: SigLipVisionConfiguration):
        super(SigLipVisionModel, self).__init__()
        self.configuration = configuration
        self.vision_encoder = ViT()


@dataclass
class ImageTokenizerConfiguration(nn.Module):
    in_channels: int = 3
    patch_size: int = 16
    embedding_size: int = 1_024


class ImageTokenizer(nn.Module):
    def __init__(self, configuration: ImageTokenizerConfiguration):
        super(ImageTokenizer, self).__init__()

        # Q: what is the relationship between the token dimensions and the hidden dimension?
        # A: they're the same. (ViT has only one hidden dim throughout)
        self.conv2d = nn.Conv2d(
            in_channels=configuration.in_channels,
            out_channels=configuration.embedding_size,
            kernel_size=configuration.patch_size,
            stride=configuration.patch_size,
            padding="valid",
        )

    def forward(self, imgs_bchw: torch.Tensor) -> torch.Tensor:
        # Returns shape [b, p, d] where d = embedding_size
        feature_maps_bdphpw = self.conv2d(
            imgs_bchw
        )  # where d = embedding dim and ph/pw = the number of patches in both directions
        return einops.rearrange(feature_maps_bdphpw, "b d ph pw -> b (ph pw) d")


def get_initial_positional_embeddings(
    num_patches: int, embedding_size: int
) -> torch.Tensor:
    return nn.Parameter(torch.randn(num_patches, embedding_size))


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()


class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()


class ViT(nn.Module):
    def __init__(self, configuration: ViTConfiguration):
        super(ViT, self).__init__()
        self.configuration = configuration

        patch_embedder_configuration = ImageTokenizerConfiguration(
            in_channels=configuration.in_channels,
            embedding_size=configuration.hidden_size,
            patch_size=configuration.patch_size,
        )
        self.patch_embedder = ImageTokenizer(patch_embedder_configuration)
        # Q: is this really the right way of writing/initialising the positional embedding?
        self.positional_embeddings = get_initial_positional_embeddings(
            num_patches=configuration.num_patches,
            embedding_size=configuration.hidden_size,
        )
        self.encoder = TransformerEncoder()
        self.classification_head = ClassificationHead()

    def forward(
        self, imgs_pixels_bchw: torch.Tensor
    ) -> torch.Tensor:  # img_pixels or imgs_pixels ?
        patches_nchw = break_images_into_patches(
            imgs_pixels_bchw, patch_size=self.configuration.patch_size
        )
        patches_embeddings_nd = self.patch_embedder(patches_nchw)
        patches_embeddings_nd += self.positional_embeddings

        # TODO: add the special classification token

        input_tokens_nd = patches_embeddings_nd
        output_tokens_nd = self.encoder(input_tokens_nd)

        # Q: what is this module supposed to return? Just the transformation of the classification token?
        #    the list of output embeddings?
        return output_tokens_nd
