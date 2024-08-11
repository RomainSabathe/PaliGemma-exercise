import torch

from paligemma.siglip import (
    ViT,
    ViTConfiguration,
    ImageTokenizer,
    ImageTokenizerConfiguration,
    LayerNorm,
)

batch_size = 2
height = 224
width = 224
channels = 3
patch_size = 16
num_patches = 196


def test_inference_vit():
    conf = ViTConfiguration()
    model = ViT(conf)

    imgs = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    y = model(imgs)

    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.float32


def test_inference_image_tokenizer():
    embedding_size = 3
    config = ImageTokenizerConfiguration(
        in_channels=channels, patch_size=patch_size, embedding_size=embedding_size
    )
    image_tokenizer = ImageTokenizer(config)
    imgs = torch.randn(batch_size, channels, height, width)

    imgs_tokens = image_tokenizer(imgs)

    assert len(imgs_tokens.shape) == 3
    assert imgs_tokens.shape[0] == batch_size
    assert imgs_tokens.shape[1] == num_patches
    assert imgs_tokens.shape[2] == embedding_size


def test_inference_layer_norm():
    layer_norm = LayerNorm(sequence_length=5, hidden_dim=10)
    x_bnd = torch.randn(batch_size, 5, 10)
    y_bnd = layer_norm(x_bnd)

    assert y_bnd.shape == x_bnd.shape
    assert y_bnd.dtype == y_bnd.dtype
