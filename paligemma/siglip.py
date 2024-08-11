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


@dataclass
class SigLipVisionConfiguration:
    encoder_configuration: ViTConfiguration


class SigLipVisionModel(nn.Module):
    def __init__(self, configuration: SigLipVisionConfiguration):
        super(SigLipVisionModel, self).__init__()
        self.configuration = configuration
        self.vision_encoder = ViT()


@dataclass
class PatchEmbedderConfiguration(nn.Module):
    in_channels: int = 3
    patch_size: int = 3
    embedding_size: int = 1_024


class PatchEmbedder(nn.Module):
    def __init__(self, configuration: PatchEmbedderConfiguration):
        super(PatchEmbedder, self).__init__()

        # Q: what is the relationship between the token dimensions and the hidden dimension?
        self.conv2d = nn.Conv2d(
            in_channels=configuration.in_channels,
            out_channels=configuration.embedding_size,
            kernel_size=configuration.patch_size,
            stride=configuration.patch_size,
            padding="valid",
        )

    def forward(self, patches_nchw: torch.Tensor) -> torch.Tensor:
        # Returns shape [n, d] where d = embedding_size
        return self.conv2d(patches_nchw)


class PositionalEmbedder(nn.Module):
    def __init__(self):
        super(PositionalEmbedder, self).__init__()

    def initial_positional_embeddings(self):
        pass


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

        patch_embedder_configuration = PatchEmbedderConfiguration(
            in_channels=configuration.in_channels,
            embedding_size=configuration.hidden_size,
            patch_size=configuration.patch_size,
        )
        self.patch_embedder = PatchEmbedder(patch_embedder_configuration)
        # Q: is this really the right way of writing/initialising the positional embedding?
        self.positional_embeddings = (
            PositionalEmbedder().initial_positional_embeddings()
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

        input_tokens_nd = patches_embeddings_nd
        output_tokens_nd = self.encoder(input_tokens_nd)

        # TODO: add the special classification token

        # Q: what is this module supposed to return? Just the transformation of the classification token?
        #    the list of output embeddings?
        return output_tokens_nd


def break_images_into_patches(imgs_bchw: torch.Tensor, patch_size: int) -> torch.Tensor:
    # Q: is there a better way of spitting the image into patches?
    return einops.rearrange(
        imgs_bchw,
        "b c (h ph) (w pw) -> (b h w) c ph pw",
        ph=patch_size,
        pw=patch_size,
    )
