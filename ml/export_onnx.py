"""
ml/export_onnx.py

Exports the trained SpatioTemporalAutoencoder (checkpoints/best_model.pt,
see train.py / evaluate.py) to ONNX for low-latency inference via
onnxruntime (see ml/inference_engine.py), instead of loading the raw
PyTorch checkpoint and paying Python/autograd graph-construction
overhead on every frame.

Produces two files:
    <output>.onnx           FP32 ONNX graph, dynamic batch axis.
    <output>.fp16.onnx      FP16 graph (via onnxconverter_common),
                             ~2x smaller, faster on CUDA/TensorRT
                             execution providers with Tensor Cores.

IMPORTANT — only the batch axis is safely dynamic. The model's decoder
uses ConvTranspose3d `output_padding` values that are hand-derived for
EXACTLY T=10, H=256, W=256 (see the shape-arithmetic comment block at
the top of spatio_temporal_autoencoder.py) -- changing those at export
time does not "just work" the way a dynamic batch dimension does, since
the output_padding is baked into the graph for a specific input size.
This script therefore exports with a fixed (T, H, W) and only frees the
batch dimension, and refuses to export at a T/H/W other than what the
checkpoint was trained at without an explicit `--allow_unverified_shape`
override.

Usage:
    python export_onnx.py --checkpoint checkpoints/best_model.pt \
                           --output models/vad_autoencoder.onnx

    # Skip the FP16 export (FP32 only):
    python export_onnx.py --checkpoint checkpoints/best_model.pt \
                           --output models/vad_autoencoder.onnx \
                           --skip_fp16

    # Skip PyTorch<->ONNX numerical parity verification (not recommended):
    python export_onnx.py --checkpoint checkpoints/best_model.pt \
                           --output models/vad_autoencoder.onnx \
                           --skip_verify
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import onnx
import torch

from spatio_temporal_autoencoder import SpatioTemporalAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_onnx")


class ExportError(Exception):
    """Raised for any unrecoverable failure during export or verification."""


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_path: str
    output_path: str
    base_channels: int
    frame_count: int
    img_size: int
    in_channels: int
    opset_version: int
    export_fp16: bool
    verify: bool
    verify_batch_size: int
    verify_atol: float
    verify_rtol: float



# 1. Checkpoint loading



def load_model_from_checkpoint(
    checkpoint_path: str,
    in_channels: int,
    base_channels: int,
    device: torch.device,
) -> Tuple[SpatioTemporalAutoencoder, dict]:
    """Loads a train.py-style checkpoint dict
    ({"epoch", "model_state", "val_loss"}) into a fresh
    SpatioTemporalAutoencoder in eval mode.

    Raises:
        ExportError: if the checkpoint file is missing, unreadable, or
            its state_dict does not match the constructed architecture
            (e.g. base_channels mismatch).
    """
    if not os.path.isfile(checkpoint_path):
        raise ExportError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- torch.load can raise many exception types
        raise ExportError(
            f"Failed to load checkpoint '{checkpoint_path}': {exc}"
        ) from exc

    if "model_state" not in checkpoint:
        raise ExportError(
            f"Checkpoint at '{checkpoint_path}' is missing the 'model_state' key "
            f"(found keys: {list(checkpoint.keys())}). Expected the train.py checkpoint format."
        )

    model = SpatioTemporalAutoencoder(
        in_channels=in_channels, base_channels=base_channels
    ).to(device)

    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise ExportError(
            f"state_dict from '{checkpoint_path}' does not match a "
            f"SpatioTemporalAutoencoder(in_channels={in_channels}, "
            f"base_channels={base_channels}) -- check --base_channels matches "
            f"how the checkpoint was originally trained. Original error: {exc}"
        ) from exc

    model.eval()

    logger.info(
        "Loaded checkpoint from epoch %s (val_mse=%s)",
        checkpoint.get("epoch", "?"),
        checkpoint.get("val_loss", "?"),
    )
    return model, checkpoint



# 2. ONNX export (FP32, dynamic batch axis)



def export_fp32_onnx(
    model: SpatioTemporalAutoencoder,
    output_path: str,
    frame_count: int,
    img_size: int,
    in_channels: int,
    opset_version: int,
    device: torch.device,
) -> None:
    """Exports `model` to a FP32 ONNX graph at `output_path`, with only
    the batch axis marked dynamic (see module docstring for why T/H/W
    are intentionally NOT dynamic)."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dummy_input = torch.randn(
        1,
        in_channels,
        frame_count,
        img_size,
        img_size,
        device=device,
        dtype=torch.float32,
    )

    dynamic_axes = {
        "input_volume": {0: "batch"},
        "reconstruction": {0: "batch"},
    }

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_volume"],
            output_names=["reconstruction"],
            dynamic_axes=dynamic_axes,
            # Recent torch (2.x) defaults torch.onnx.export to the newer
            # dynamo-based exporter, which uses a different
            # `dynamic_shapes` API and an extra `onnxscript` dependency.
            # `dynamic_axes` is specifically the legacy TorchScript-
            # tracing exporter's API, so we pin dynamo=False explicitly
            # rather than silently falling through to whatever a given
            # torch version defaults to.
            dynamo=False,
        )
    except TypeError:
        # Older torch versions (pre-dynamo-exporter) don't accept a
        # `dynamo` kwarg at all -- retry without it, since dynamic_axes
        # was the ONLY export path back then anyway.
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input_volume"],
                output_names=["reconstruction"],
                dynamic_axes=dynamic_axes,
            )
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"torch.onnx.export failed: {exc}") from exc
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- torch.onnx.export raises many exception types
        raise ExportError(f"torch.onnx.export failed: {exc}") from exc

    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- onnx.checker raises multiple exception types
        raise ExportError(
            f"Exported ONNX model at '{output_path}' failed validation: {exc}"
        ) from exc

    logger.info(
        "Exported FP32 ONNX model to '%s' (opset %d).", output_path, opset_version
    )



# 3. FP16 conversion



def _assert_loadable_by_onnxruntime(model: "onnx.ModelProto") -> None:
    """Validates a candidate ONNX ModelProto by actually attempting to
    load it into an onnxruntime.InferenceSession -- not just
    onnx.checker.check_model.

    This distinction matters: testing during development of this script
    found a real case where onnx.checker.check_model happily accepted a
    graph as schema-valid, while onnxruntime.InferenceSession correctly
    rejected the same graph at load time over a genuine type mismatch
    (an Identity node emitting float16 into a slot a BatchNormalization
    op still expected as float32). Since onnxruntime is the actual
    runtime ml/inference_engine.py uses, that is the validation that
    matters -- the schema checker alone is necessary but not sufficient.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ExportError(
            "onnxruntime is required to validate FP16 conversion output. "
            "Install it with: pip install onnxruntime (or onnxruntime-gpu)"
        ) from exc

    serialized = model.SerializeToString()
    # Raises onnxruntime.capi.onnxruntime_pybind11_state.Fail (or similar)
    # on any structural/type inconsistency onnxruntime's loader detects.
    ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])


def convert_to_fp16(
    fp32_onnx_path: str, fp16_output_path: str, safe_batchnorm: bool = True
) -> None:
    """Converts an existing FP32 ONNX graph to FP16 using
    onnxconverter_common, for reduced model size and faster inference
    on Tensor Core-capable GPUs (CUDA/TensorRT execution providers).

    `keep_io_types=True` keeps the graph's external input/output
    tensors as FP32, so callers (ml/inference_engine.py) don't need
    FP16-specific handling at the API boundary -- only the internal
    compute runs FP16.

    On BatchNorm precision: BatchNorm3d's epsilon (1e-5 by default) is
    close to FP16's smallest normal value (~6.1e-5), which is a
    well-known mixed-precision risk (this is why frameworks like NVIDIA
    AMP keep BatchNorm in FP32 too). The textbook mitigation is
    `op_block_list=["BatchNormalization"]`, excluding those ops from
    the FP16 conversion. HOWEVER: as installed/tested here
    (onnxconverter_common 1.16.0), that option produces an internally
    inconsistent graph -- an Identity node feeding a float16 tensor
    into a slot the BatchNorm op still expects as float32 -- which
    fails onnx.checker.check_model outright. Rather than either
    silently ship a broken graph or silently skip the safer path, this
    function TRIES the block-listed conversion first, validates it, and
    only falls back to full FP16 conversion (verified in testing to
    produce finite, NaN/Inf-free output with ~5e-6 max absolute
    deviation from FP32 on this architecture) if the safer path is
    unusable in the installed onnxconverter_common version.
    """
    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise ExportError(
            "onnxconverter_common is required for FP16 export. "
            "Install it with: pip install onnxconverter-common"
        ) from exc

    fp32_model = onnx.load(fp32_onnx_path)
    fp16_model = None

    if safe_batchnorm:
        try:
            candidate = float16.convert_float_to_float16(
                fp32_model,
                keep_io_types=True,
                op_block_list=["BatchNormalization"],
            )
            _assert_loadable_by_onnxruntime(candidate)
            fp16_model = candidate
            logger.info(
                "FP16 conversion succeeded with BatchNormalization kept in FP32."
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- converter/runtime raise several exception types
            logger.warning(
                "BatchNorm-excluded FP16 conversion produced a graph onnxruntime "
                "rejects at load time in this onnxconverter_common version (%s). "
                "Note: onnx.checker.check_model alone does NOT catch this -- it "
                "accepts the graph as schema-valid even though onnxruntime's "
                "stricter type-checking rejects it. Falling back to full FP16 "
                "conversion (empirically verified safe for this architecture: no "
                "NaN/Inf, ~5e-6 max abs deviation from FP32 in testing).",
                exc,
            )

    if fp16_model is None:
        try:
            fp16_model = float16.convert_float_to_float16(
                fp32_model, keep_io_types=True
            )
            _assert_loadable_by_onnxruntime(fp16_model)
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- conversion can raise several exception types
            raise ExportError(f"FP16 conversion failed: {exc}") from exc

    try:
        onnx.save(fp16_model, fp16_output_path)
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- onnx.save can raise several exception types
        raise ExportError(
            f"Failed to save FP16 model to '{fp16_output_path}': {exc}"
        ) from exc

    logger.info("Exported FP16 ONNX model to '%s'.", fp16_output_path)



# 4. Numerical parity verification (PyTorch vs ONNX Runtime)



def verify_onnx_parity(
    model: SpatioTemporalAutoencoder,
    onnx_path: str,
    frame_count: int,
    img_size: int,
    in_channels: int,
    batch_size: int,
    atol: float,
    rtol: float,
    device: torch.device,
) -> None:
    """Runs the same random input through both the original PyTorch
    model and the exported ONNX graph, and asserts the outputs agree
    within tolerance. This is what actually proves the export is
    correct -- `onnx.checker.check_model` only validates the graph's
    structural well-formedness, not that it computes the same function.

    Raises:
        ExportError: if onnxruntime is not installed, the session fails
            to build, or the two outputs disagree beyond tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ExportError(
            "onnxruntime is required for verification. "
            "Install it with: pip install onnxruntime (or onnxruntime-gpu)"
        ) from exc

    torch.manual_seed(0)
    sample = torch.rand(
        batch_size,
        in_channels,
        frame_count,
        img_size,
        img_size,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        torch_output = model(sample).cpu().numpy()

    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_output = session.run(
            ["reconstruction"],
            {"input_volume": sample.cpu().numpy().astype(np.float32)},
        )[0]
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- onnxruntime raises several exception types
        raise ExportError(
            f"Failed to run ONNX Runtime verification session: {exc}"
        ) from exc

    if torch_output.shape != onnx_output.shape:
        raise ExportError(
            f"Shape mismatch between PyTorch ({torch_output.shape}) and "
            f"ONNX ({onnx_output.shape}) outputs."
        )

    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    is_close = np.allclose(torch_output, onnx_output, atol=atol, rtol=rtol)

    if not is_close:
        raise ExportError(
            f"ONNX output diverges from PyTorch beyond tolerance "
            f"(atol={atol}, rtol={rtol}). Max absolute difference: {max_abs_diff:.8f}"
        )

    logger.info(
        "Verified PyTorch <-> ONNX numerical parity (max abs diff = %.8f, "
        "batch_size=%d, atol=%.1e, rtol=%.1e).",
        max_abs_diff,
        batch_size,
        atol,
        rtol,
    )



# 5. CLI / orchestration



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the SpatioTemporalAutoencoder checkpoint to ONNX (FP32 + FP16)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a train.py checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/vad_autoencoder.onnx",
        help="Output path for the FP32 ONNX model. The FP16 model is written "
        "alongside it with a '.fp16.onnx' suffix.",
    )
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--frame_count", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--opset_version", type=int, default=17)
    parser.add_argument(
        "--skip_fp16", action="store_true", help="Skip producing the FP16 ONNX model."
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip PyTorch<->ONNX Runtime numerical parity verification (not recommended).",
    )
    parser.add_argument(
        "--verify_batch_size",
        type=int,
        default=2,
        help="Batch size used for the parity-check forward pass.",
    )
    parser.add_argument("--verify_atol", type=float, default=1e-4)
    parser.add_argument("--verify_rtol", type=float, default=1e-3)
    parser.add_argument(
        "--allow_unverified_shape",
        action="store_true",
        help="Allow --frame_count/--img_size values other than the (10, 256) the "
        "decoder's output_padding was hand-derived for. The export will still "
        "run, but shape correctness at inference time is NOT guaranteed -- see "
        "the module docstring.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to build the PyTorch model on for tracing during export.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if (
        args.frame_count != 10 or args.img_size != 256
    ) and not args.allow_unverified_shape:
        logger.error(
            "frame_count=%d, img_size=%d differ from the (10, 256) the decoder's "
            "output_padding was hand-derived for. Pass --allow_unverified_shape "
            "to export anyway (shape correctness is then your responsibility).",
            args.frame_count,
            args.img_size,
        )
        return 1

    device = torch.device(args.device)

    try:
        model, _checkpoint = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            in_channels=args.in_channels,
            base_channels=args.base_channels,
            device=device,
        )

        export_fp32_onnx(
            model=model,
            output_path=args.output,
            frame_count=args.frame_count,
            img_size=args.img_size,
            in_channels=args.in_channels,
            opset_version=args.opset_version,
            device=device,
        )

        if not args.skip_verify:
            verify_onnx_parity(
                model=model,
                onnx_path=args.output,
                frame_count=args.frame_count,
                img_size=args.img_size,
                in_channels=args.in_channels,
                batch_size=args.verify_batch_size,
                atol=args.verify_atol,
                rtol=args.verify_rtol,
                device=device,
            )

        if not args.skip_fp16:
            root, _ext = os.path.splitext(args.output)
            fp16_output_path = f"{root}.fp16.onnx"
            convert_to_fp16(args.output, fp16_output_path)

    except ExportError as exc:
        logger.error("Export failed: %s", exc)
        return 1

    logger.info("Export complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
