"""

PyTorch Spatial-Temporal Core for the self-supervised /
unsupervised Video Anomaly Detection (VAD) engine.

This module is designed to sit alongside the existing YOLOv8 rule-based
logging pipeline and the FAISS/Gemini RAG pipeline.
It does NOT depend on either of them — it is a standalone, self-supervised
reconstruction model. The intended training signal is: train only on
"normal" behavior clips, minimize reconstruction error (MSE), and at
inference time flag clips whose reconstruction error exceeds a threshold
as behavioral anomalies (fighting, running, panic movement, etc).

"""

from __future__ import annotations

import torch
import torch.nn as nn



# SHAPE ARITHMETIC (documented once here, referenced inline on every layer)

# For a single spatial/temporal dimension, Conv3d output size is:
#
#     out = floor((in + 2*pad - kernel) / stride) + 1
#
# and the ConvTranspose3d (deconvolution) inverse is:
#
#     out = (in - 1)*stride - 2*pad + kernel + output_padding
#
# We use kernel=3 and padding=1 everywhere, so a stride-1 layer never
# changes a dimension's size, and a stride-2 layer exactly halves it
# (integer floor division). This lets us design a symmetric encoder/decoder
# where every downsampling Conv3d has a mirrored ConvTranspose3d that
# restores the exact original size, by choosing output_padding to correct
# for the floor() lost in the encoder when a dimension is odd (e.g. T=5 -> 3
# on the way down needs output_padding to go 3 -> 5 on the way up).
#
# Input volume: (B, C=1, T=10, H=256, W=256)
#
# ENCODER (spatial stride 2 on every layer; temporal stride 2 only on the
# two middle layers, so we compress space faster than time — behavioral
# anomalies like running/fighting need more temporal resolution preserved
# than pure spatial texture):
#
#   Layer 1: Conv3d stride=(1,2,2), pad=1, kernel=3
#       T: floor((10 + 2 - 3)/1) + 1 = 10        (unchanged, stride 1)
#       H: floor((256 + 2 - 3)/2) + 1 = 128       (halved, stride 2)
#       W: floor((256 + 2 - 3)/2) + 1 = 128       (halved, stride 2)
#       -> (B, 16, 10, 128, 128)
#
#   Layer 2: Conv3d stride=(2,2,2), pad=1, kernel=3
#       T: floor((10 + 2 - 3)/2) + 1 = 5
#       H: floor((128 + 2 - 3)/2) + 1 = 64
#       W: floor((128 + 2 - 3)/2) + 1 = 64
#       -> (B, 32, 5, 64, 64)
#
#   Layer 3: Conv3d stride=(2,2,2), pad=1, kernel=3
#       T: floor((5 + 2 - 3)/2) + 1 = 3
#       H: floor((64 + 2 - 3)/2) + 1 = 32
#       W: floor((64 + 2 - 3)/2) + 1 = 32
#       -> (B, 64, 3, 32, 32)
#
#   Layer 4 (bottleneck): Conv3d stride=(1,2,2), pad=1, kernel=3
#       T: floor((3 + 2 - 3)/1) + 1 = 3           (unchanged, stride 1)
#       H: floor((32 + 2 - 3)/2) + 1 = 16
#       W: floor((32 + 2 - 3)/2) + 1 = 16
#       -> (B, 128, 3, 16, 16)   <-- latent spatio-temporal representation
#
# DECODER (exact mirror image, ConvTranspose3d, same kernel/pad, with
# output_padding chosen per the formula above to hit the exact encoder
# shape at each stage):
#
#   Layer 1: ConvTranspose3d stride=(1,2,2), pad=1, kernel=3, out_pad=(0,1,1)
#       T: (3-1)*1 - 2 + 3 + 0 = 3
#       H: (16-1)*2 - 2 + 3 + 1 = 32
#       W: (16-1)*2 - 2 + 3 + 1 = 32
#       -> (B, 64, 3, 32, 32)
#
#   Layer 2: ConvTranspose3d stride=(2,2,2), pad=1, kernel=3, out_pad=(0,1,1)
#       T: (3-1)*2 - 2 + 3 + 0 = 5
#       H: (32-1)*2 - 2 + 3 + 1 = 64
#       W: (32-1)*2 - 2 + 3 + 1 = 64
#       -> (B, 32, 5, 64, 64)
#
#   Layer 3: ConvTranspose3d stride=(2,2,2), pad=1, kernel=3, out_pad=(1,1,1)
#       T: (5-1)*2 - 2 + 3 + 1 = 10
#       H: (64-1)*2 - 2 + 3 + 1 = 128
#       W: (64-1)*2 - 2 + 3 + 1 = 128
#       -> (B, 16, 10, 128, 128)
#
#   Layer 4: ConvTranspose3d stride=(1,2,2), pad=1, kernel=3, out_pad=(0,1,1)
#       T: (10-1)*1 - 2 + 3 + 0 = 10
#       H: (128-1)*2 - 2 + 3 + 1 = 256
#       W: (128-1)*2 - 2 + 3 + 1 = 256
#       -> (B, 1, 10, 256, 256)   <-- Sigmoid, reconstructed volume
#
# The reconstructed tensor is guaranteed bit-for-bit shape-identical to the
# input for the design resolution (T=10, H=256, W=256). 



class SpatioTemporalAutoencoder(nn.Module):
    """
    3D Convolutional Autoencoder for unsupervised video anomaly detection.

    Input / output tensor shape: (B, C=1, T=10, H=256, W=256)

    Trained with a reconstruction loss (e.g. nn.MSELoss) on normal-behavior
    clips only. At inference time, per-clip reconstruction error is used as
    an anomaly score — high error indicates the model has not learned to
    reconstruct that spatio-temporal pattern, i.e. it is out-of-distribution
    behavior (fighting, running, falling, panic movement, etc).
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

       
        # Encoder
       
        self.encoder = nn.Sequential(
            # Layer 1: (1,10,256,256) -> (c1,10,128,128)
            nn.Conv3d(in_channels, c1, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(c1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: (c1,10,128,128) -> (c2,5,64,64)
            nn.Conv3d(c1, c2, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(c2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: (c2,5,64,64) -> (c3,3,32,32)
            nn.Conv3d(c2, c3, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(c3),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4 (bottleneck): (c3,3,32,32) -> (c4,3,16,16)
            nn.Conv3d(c3, c4, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(c4),
            nn.LeakyReLU(0.2, inplace=True),
        )

       
        # Decoder (exact mirror — see shape derivation above)
       
        self.decoder = nn.Sequential(
            # Layer 1: (c4,3,16,16) -> (c3,3,32,32)
            nn.ConvTranspose3d(
                c4, c3, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(c3),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: (c3,3,32,32) -> (c2,5,64,64)
            nn.ConvTranspose3d(
                c3, c2, kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(c2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: (c2,5,64,64) -> (c1,10,128,128)
            nn.ConvTranspose3d(
                c2, c1, kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(c1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: (c1,10,128,128) -> (in_channels,10,256,256)
            nn.ConvTranspose3d(
                c1, in_channels, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) with T=10, H=256, W=256
        returns: reconstructed tensor of identical shape
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    @staticmethod
    def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Per-sample anomaly score. Returns a (B,) tensor of mean squared
        reconstruction error per clip when reduction='mean' (default),
        or the raw per-element error map when reduction='none'.
        """
        error_map = (x - x_hat) ** 2
        if reduction == "none":
            return error_map
        # mean over C, T, H, W -> one scalar score per clip in the batch
        return error_map.mean(dim=(1, 2, 3, 4))



# Shape-validation main block

if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalAutoencoder(in_channels=1, base_channels=16).to(device)

    B, C, T, H, W = 2, 1, 10, 256, 256
    dummy = torch.rand(B, C, T, H, W, device=device)

    print(f"Input shape:  {tuple(dummy.shape)}")

    # Sanity-check the bottleneck shape matches the documented derivation.
    with torch.no_grad():
        latent = model.encode(dummy)
        print(f"Latent shape: {tuple(latent.shape)}  (expected (B, 128, 3, 16, 16))")
        assert latent.shape == (B, 128, 3, 16, 16), "Bottleneck shape mismatch vs design spec."

        reconstruction = model.decoder(latent)
        print(f"Output shape: {tuple(reconstruction.shape)}")

    assert reconstruction.shape == dummy.shape, (
        f"Shape mismatch: input {tuple(dummy.shape)} vs "
        f"reconstruction {tuple(reconstruction.shape)}"
    )
    assert reconstruction.min() >= 0.0 and reconstruction.max() <= 1.0, (
        "Sigmoid output out of [0, 1] range."
    )

    # End-to-end forward pass (encoder + decoder together)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == dummy.shape

    # Anomaly-score sanity check
    scores = SpatioTemporalAutoencoder.reconstruction_error(dummy, out)
    print(f"Per-clip reconstruction error (anomaly score): {scores.tolist()}")
    assert scores.shape == (B,)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")
    print("All shape-validation checks passed.")
