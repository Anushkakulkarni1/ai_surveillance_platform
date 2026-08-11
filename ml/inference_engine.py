"""
ml/inference_engine.py

High-performance ONNX Runtime inference wrapper for the exported
SpatioTemporalAutoencoder (see export_onnx.py). Replaces loading the
raw PyTorch checkpoint (checkpoints/best_model.pt) directly at
inference time -- which pays Python/autograd graph-construction
overhead on every call -- with a pre-built, execution-provider-
optimized ONNX Runtime session.

Typical usage (mirrors LiveVADScorer in live_inference.py, but backed
by ONNX Runtime instead of a raw PyTorch nn.Module):

    engine = ONNXInferenceEngine("models/vad_autoencoder.onnx")
    scores = engine.predict_batch(frame_stack)   # frame_stack: (B, T, H, W)
    # scores: (B,) normalized-scale reconstruction error per clip
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover -- torch is optional for this module
    _TORCH_AVAILABLE = False

import onnxruntime as ort

logger = logging.getLogger("ml.inference_engine")



# Exceptions



class InferenceEngineError(Exception):
    """Base class for all errors raised by ONNXInferenceEngine."""


class ModelLoadError(InferenceEngineError):
    """Raised when the ONNX model file cannot be loaded into a session."""


class InvalidInputShapeError(InferenceEngineError):
    """Raised when predict_batch() receives an array of the wrong rank/shape."""


class InferenceExecutionError(InferenceEngineError):
    """Raised when the ONNX Runtime session itself fails during run()."""



# Result container



@dataclass(frozen=True)
class InferenceResult:
    """Structured output of a single predict_batch() call."""

    scores: np.ndarray  # shape (B,), float32, per-clip reconstruction error
    reconstructions: Optional[
        np.ndarray
    ]  # shape (B, C, T, H, W) if requested, else None
    latency_ms: float  # wall-clock time for the session.run() call only
    provider_used: str  # the execution provider onnxruntime actually selected



# Engine



class ONNXInferenceEngine:
    """Wraps an onnxruntime.InferenceSession for the SpatioTemporalAutoencoder,
    with automatic CUDA -> CPU execution-provider fallback, input
    normalization, batching, and reconstruction-error scoring built in.
    """

    _EXPECTED_INPUT_RANK = 5  # (B, C, T, H, W)

    def __init__(
        self,
        model_path: str,
        frame_count: int = 10,
        img_size: int = 256,
        in_channels: int = 1,
        prefer_cuda: bool = True,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
        graph_optimization_level: "ort.GraphOptimizationLevel" = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ),
        warmup: bool = True,
        input_name: str = "input_volume",
        output_name: str = "reconstruction",
    ) -> None:
        if frame_count <= 0 or img_size <= 0 or in_channels <= 0:
            raise ValueError(
                f"frame_count, img_size, and in_channels must all be positive; got "
                f"frame_count={frame_count}, img_size={img_size}, in_channels={in_channels}"
            )

        self.frame_count = frame_count
        self.img_size = img_size
        self.in_channels = in_channels
        self.input_name = input_name
        self.output_name = output_name

        self._session, self._provider_used = self._build_session(
            model_path=model_path,
            prefer_cuda=prefer_cuda,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
            graph_optimization_level=graph_optimization_level,
        )

        self._input_dtype = self._resolve_onnx_input_dtype()

        logger.info(
            "ONNXInferenceEngine ready: model='%s', provider='%s', input_dtype=%s",
            model_path,
            self._provider_used,
            self._input_dtype,
        )

        if warmup:
            self._warmup()

    
    # Construction helpers
    

    @staticmethod
    def _build_session(
        model_path: str,
        prefer_cuda: bool,
        intra_op_num_threads: Optional[int],
        inter_op_num_threads: Optional[int],
        graph_optimization_level: "ort.GraphOptimizationLevel",
    ) -> "tuple[ort.InferenceSession, str]":
        """Builds the InferenceSession, preferring CUDAExecutionProvider
        when available and requested, transparently falling back to
        CPUExecutionProvider otherwise -- the caller never needs to know
        which one actually ran.
        """
        available_providers = ort.get_available_providers()

        if prefer_cuda and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            if prefer_cuda:
                logger.warning(
                    "prefer_cuda=True but CUDAExecutionProvider is not available "
                    "in this onnxruntime build/environment (available: %s). "
                    "Falling back to CPUExecutionProvider.",
                    available_providers,
                )
            providers = ["CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = graph_optimization_level
        if intra_op_num_threads is not None:
            session_options.intra_op_num_threads = intra_op_num_threads
        if inter_op_num_threads is not None:
            session_options.inter_op_num_threads = inter_op_num_threads

        try:
            session = ort.InferenceSession(
                model_path, sess_options=session_options, providers=providers
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- onnxruntime raises several exception types
            raise ModelLoadError(
                f"Failed to load ONNX model from '{model_path}': {exc}"
            ) from exc

        actual_providers = session.get_providers()
        provider_used = actual_providers[0] if actual_providers else "unknown"

        if prefer_cuda and provider_used != "CUDAExecutionProvider":
            logger.warning(
                "Requested CUDAExecutionProvider but the session is actually running "
                "on '%s'. Check your CUDA/cuDNN install and the onnxruntime-gpu "
                "package (a plain 'onnxruntime' install has no CUDA provider at all).",
                provider_used,
            )

        return session, provider_used

    def _resolve_onnx_input_dtype(self) -> np.dtype:
        """Inspects the session's declared input tensor type so
        predict_batch() can automatically cast to whatever the graph
        actually expects (float32 for an FP32 export, or a
        keep_io_types=True FP16 export which -- per export_onnx.py --
        also expects float32 at the boundary)."""
        onnx_type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
        }
        session_inputs = self._session.get_inputs()
        matching = [inp for inp in session_inputs if inp.name == self.input_name]
        if not matching:
            available = [inp.name for inp in session_inputs]
            raise ModelLoadError(
                f"Input '{self.input_name}' not found in the loaded ONNX model. "
                f"Available inputs: {available}"
            )
        onnx_dtype_str = matching[0].type
        if onnx_dtype_str not in onnx_type_map:
            raise ModelLoadError(
                f"Unsupported ONNX input dtype '{onnx_dtype_str}' for input "
                f"'{self.input_name}'. Supported: {list(onnx_type_map.keys())}"
            )
        return np.dtype(onnx_type_map[onnx_dtype_str])

    def _warmup(self) -> None:
        """Runs one dummy inference at construction time so the first
        real predict_batch() call doesn't pay CUDA kernel
        compilation/allocator warmup cost (a well-known
        onnxruntime-on-GPU latency spike on the very first call)."""
        try:
            dummy = np.zeros(
                (1, self.in_channels, self.frame_count, self.img_size, self.img_size),
                dtype=self._input_dtype,
            )
            self._session.run([self.output_name], {self.input_name: dummy})
            logger.info("Warmup inference completed.")
        except Exception as exc:  # noqa: BLE001 -- warmup failures shouldn't be fatal
            logger.warning("Warmup inference failed (continuing anyway): %s", exc)

    
    # Input normalization / coercion
    

    def _coerce_to_numpy(self, frames: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        if _TORCH_AVAILABLE and isinstance(frames, torch.Tensor):
            return frames.detach().cpu().numpy()
        if isinstance(frames, np.ndarray):
            return frames
        raise InvalidInputShapeError(
            f"frames must be a numpy.ndarray or torch.Tensor, got {type(frames)!r}"
        )

    def _validate_and_reshape(self, frames: np.ndarray) -> np.ndarray:
        """Accepts either:
            (B, T, H, W)      -- grayscale, channel dim implicit (in_channels=1)
            (B, C, T, H, W)   -- already channel-explicit
        and returns a validated (B, C, T, H, W) array. Rejects anything
        else with a precise error rather than letting a shape mismatch
        surface as an opaque onnxruntime error deep inside session.run().
        """
        if frames.ndim == self._EXPECTED_INPUT_RANK - 1:
            # (B, T, H, W) -> (B, 1, T, H, W); only valid when the model
            # itself expects a single channel.
            if self.in_channels != 1:
                raise InvalidInputShapeError(
                    f"Received a 4D array (B, T, H, W) but this engine's in_channels="
                    f"{self.in_channels} (!= 1) -- pass an explicit channel dimension: "
                    f"(B, {self.in_channels}, T, H, W)."
                )
            frames = frames[:, np.newaxis, ...]
        elif frames.ndim != self._EXPECTED_INPUT_RANK:
            raise InvalidInputShapeError(
                f"Expected a 4D (B, T, H, W) or 5D (B, C, T, H, W) array, got "
                f"{frames.ndim}D array with shape {frames.shape}."
            )

        expected_shape_tail = (
            self.in_channels,
            self.frame_count,
            self.img_size,
            self.img_size,
        )
        if tuple(frames.shape[1:]) != expected_shape_tail:
            raise InvalidInputShapeError(
                f"Expected shape (B, {', '.join(map(str, expected_shape_tail))}), "
                f"got {tuple(frames.shape)}."
            )

        return frames

    def _normalize(self, frames: np.ndarray, already_normalized: bool) -> np.ndarray:
        """Scales raw pixel values into [0, 1], matching the training-time
        preprocessing (`img.astype(np.float32) / 255.0`, see
        data_pipeline.py's VideoClipDataset / live_inference.py's
        LiveVADScorer.push_frame). If the caller has already normalized
        the frames (already_normalized=True), this is a no-op beyond the
        dtype cast.
        """
        frames = frames.astype(np.float32, copy=False)
        if already_normalized:
            return frames
        # Heuristic guard: if the values are already in [0, 1] despite
        # already_normalized=False, dividing by 255 again would be
        # silently wrong -- fail loudly instead of producing a
        # near-all-zero volume that would look like "everything is
        # normal" to the anomaly detector.
        max_value = float(frames.max()) if frames.size > 0 else 0.0
        if max_value <= 1.0 and max_value > 0.0:
            raise InvalidInputShapeError(
                f"already_normalized=False but the input's max value is {max_value:.4f} "
                f"(<= 1.0), suggesting it is already normalized. Pass "
                f"already_normalized=True to skip the /255.0 scaling, or verify the "
                f"input is genuinely raw 0-255 pixel data."
            )
        return frames / 255.0

    
    # Public inference API
    

    def predict_batch(
        self,
        frames: Union[np.ndarray, "torch.Tensor"],
        already_normalized: bool = False,
        return_reconstructions: bool = False,
        max_sub_batch_size: Optional[int] = None,
    ) -> InferenceResult:
        """Runs a batch of clips through the ONNX model and returns
        per-clip reconstruction error scores.

        Args:
            frames: (B, T, H, W) or (B, C, T, H, W) array/tensor of raw
                grayscale pixel values in [0, 255] (or already in [0, 1]
                if already_normalized=True).
            already_normalized: skip the /255.0 scaling if the caller has
                already normalized the input.
            return_reconstructions: if True, also return the full
                reconstructed volumes (memory-heavy for large batches;
                default False since most callers only need the score).
            max_sub_batch_size: if set, splits `frames` into sequential
                sub-batches of at most this size (useful to bound peak
                GPU/CPU memory for a large batch) and concatenates the
                results. If None, the whole batch is run in one
                session.run() call.

        Returns:
            InferenceResult with per-clip scores, optional
            reconstructions, measured latency, and the execution
            provider actually used.

        Raises:
            InvalidInputShapeError: on a malformed/mismatched input shape.
            InferenceExecutionError: if the underlying session.run() fails.
        """
        numpy_frames = self._coerce_to_numpy(frames)
        numpy_frames = self._validate_and_reshape(numpy_frames)
        numpy_frames = self._normalize(
            numpy_frames, already_normalized=already_normalized
        )

        batch_size = numpy_frames.shape[0]
        if batch_size == 0:
            return InferenceResult(
                scores=np.empty((0,), dtype=np.float32),
                reconstructions=(
                    np.empty(
                        (
                            0,
                            self.in_channels,
                            self.frame_count,
                            self.img_size,
                            self.img_size,
                        )
                    )
                    if return_reconstructions
                    else None
                ),
                latency_ms=0.0,
                provider_used=self._provider_used,
            )

        sub_batches: List[np.ndarray]
        if max_sub_batch_size is not None and max_sub_batch_size > 0:
            sub_batches = [
                numpy_frames[i : i + max_sub_batch_size]
                for i in range(0, batch_size, max_sub_batch_size)
            ]
        else:
            sub_batches = [numpy_frames]

        all_scores: List[np.ndarray] = []
        all_reconstructions: List[np.ndarray] = [] if return_reconstructions else []
        total_latency_ms = 0.0

        for sub_batch in sub_batches:
            onnx_input = sub_batch.astype(self._input_dtype, copy=False)

            start = time.perf_counter()
            try:
                outputs = self._session.run(
                    [self.output_name], {self.input_name: onnx_input}
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001 -- onnxruntime raises several exception types
                raise InferenceExecutionError(
                    f"ONNX Runtime session.run() failed: {exc}"
                ) from exc
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            total_latency_ms += elapsed_ms

            reconstruction = outputs[0].astype(np.float32, copy=False)

            # Mirrors SpatioTemporalAutoencoder.reconstruction_error's
            # convention exactly: mean squared error over (C, T, H, W),
            # one score per clip in the batch.
            error_map = (sub_batch.astype(np.float32, copy=False) - reconstruction) ** 2
            scores = error_map.mean(axis=(1, 2, 3, 4))

            all_scores.append(scores)
            if return_reconstructions:
                all_reconstructions.append(reconstruction)

        final_scores = np.concatenate(all_scores, axis=0).astype(np.float32)
        final_reconstructions = (
            np.concatenate(all_reconstructions, axis=0)
            if return_reconstructions
            else None
        )

        return InferenceResult(
            scores=final_scores,
            reconstructions=final_reconstructions,
            latency_ms=total_latency_ms,
            provider_used=self._provider_used,
        )

    
    # Introspection / diagnostics
    

    @property
    def provider(self) -> str:
        """The execution provider actually in use (e.g.
        'CUDAExecutionProvider' or 'CPUExecutionProvider')."""
        return self._provider_used

    @property
    def input_dtype(self) -> np.dtype:
        """The dtype the underlying ONNX graph expects at its input
        boundary (float32 even for a keep_io_types=True FP16 export)."""
        return self._input_dtype

    def get_expected_input_shape(
        self, batch_size: int = 1
    ) -> "tuple[int, int, int, int, int]":
        return (
            batch_size,
            self.in_channels,
            self.frame_count,
            self.img_size,
            self.img_size,
        )



# Convenience: normalize a raw frame sequence into the (T, H, W)
# shape predict_batch() expects, mirroring the preprocessing already
# used by data_pipeline.VideoClipDataset / live_inference.LiveVADScorer.



def frames_to_clip_array(frames: Sequence[np.ndarray], img_size: int) -> np.ndarray:
    """Stacks a sequence of raw grayscale frames (each H_orig x W_orig,
    uint8 or float) into a single (T, H, W) float32 array resized to
    (img_size, img_size), ready to be batched and passed to
    predict_batch(). Requires `cv2` (already a project dependency).

    Raises:
        InvalidInputShapeError: if `frames` is empty or contains frames
            that fail to resize (e.g. a corrupt/empty array).
    """
    if len(frames) == 0:
        raise InvalidInputShapeError("frames must contain at least one frame.")

    try:
        import cv2
    except ImportError as exc:
        raise InferenceEngineError(
            "opencv-python (cv2) is required for frames_to_clip_array(). "
            "Install it with: pip install opencv-python"
        ) from exc

    resized_frames = []
    for index, frame in enumerate(frames):
        if frame is None or frame.size == 0:
            raise InvalidInputShapeError(f"Frame at index {index} is empty/None.")
        resized = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        resized_frames.append(resized.astype(np.float32))

    return np.stack(resized_frames, axis=0)
