"""Runs live video anomaly detection using an ONNX model.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

import cv2
import httpx
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.schemas import AnomalyAlert, StreamTelemetry  # noqa: E402
from data_pipeline import describe_anomaly  # noqa: E402
from frame_buffer import BufferConfigError, SlidingWindowFrameBuffer  # noqa: E402
from inference_engine import (  # noqa: E402
    InferenceEngineError,
    InvalidInputShapeError,
    ONNXInferenceEngine,
)
from live_inference import (  # noqa: E402
    load_calibration_bounds,
    normalize_fixed,
    resolve_source,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("live_inference_onnx")






class TelemetryPublisher:
 

    def __init__(
        self,
        backend_url: str,
        request_timeout: float = 2.0,
        max_retries: int = 1,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError(f"request_timeout must be positive, got {request_timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")

        self._max_retries = max_retries
        self._client = httpx.Client(
            base_url=backend_url.rstrip("/"), timeout=request_timeout
        )
        self.consecutive_failures = 0

    def health_check(self) -> bool:
        
        try:
            response = self._client.get("/health")
            response.raise_for_status()
            body = response.json()
            return bool(body.get("redis_connected", False))
        except httpx.HTTPError as exc:
            logger.warning("Backend health check failed: %s", exc)
            return False

    def publish(self, telemetry: StreamTelemetry) -> bool:
        
        payload = telemetry.model_dump(mode="json", by_alias=False)

        attempt = 0
        while True:
            try:
                response = self._client.post("/ingest/telemetry", json=payload)
                response.raise_for_status()
                self.consecutive_failures = 0
                return True
            except httpx.HTTPError as exc:
                attempt += 1
                if attempt > self._max_retries:
                    self.consecutive_failures += 1
                    logger.warning(
                        "Failed to publish telemetry frame_id=%s after %d attempt(s): %s "
                        "(consecutive failures: %d)",
                        telemetry.frame_id,
                        attempt,
                        exc,
                        self.consecutive_failures,
                    )
                    return False

    def close(self) -> None:
        self._client.close()



# ONNX-backed live VAD scorer + publisher



class ONNXLiveVADPublisher:
    

    def __init__(
        self,
        engine: ONNXInferenceEngine,
        buffer: SlidingWindowFrameBuffer,
        publisher: TelemetryPublisher,
        calibration_low: float,
        calibration_high: float,
        zone: str,
        alert_threshold: float,
        source_name: str,
        min_publish_interval_sec: float = 2.0,
        always_publish: bool = False,
    ) -> None:
        self._engine = engine
        self._buffer = buffer
        self._publisher = publisher
        self._low = calibration_low
        self._high = calibration_high
        self._zone = zone
        self._alert_threshold = alert_threshold
        self._source_name = source_name
        self._min_publish_interval_sec = min_publish_interval_sec
        self._always_publish = always_publish

        self._frame_id = 0
        self._last_publish_time = 0.0

    def process_frame(self, bgr_frame: np.ndarray) -> Optional[dict]:
        
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            (self._buffer.img_size, self._buffer.img_size),
            interpolation=cv2.INTER_AREA,
        )
        normalized_frame = gray.astype(np.float32) / 255.0

        try:
            volume = self._buffer.push(normalized_frame)
        except BufferConfigError as exc:
            logger.error("Frame buffer rejected a frame: %s", exc)
            return None

        self._frame_id += 1

        if volume is None:
            return None  

        try:
            result = self._engine.predict_batch(volume, already_normalized=True)
        except (InvalidInputShapeError, InferenceEngineError) as exc:
            logger.error(
                "ONNX inference failed for frame_id=%d: %s", self._frame_id, exc
            )
            return None

        raw_score = float(result.scores[0])
        normalized_score = normalize_fixed(raw_score, self._low, self._high)
        is_critical = normalized_score >= self._alert_threshold
        description = describe_anomaly(
            normalized_score, zone=self._zone, base_threshold=self._alert_threshold
        )

        now = time.time()
        interval_elapsed = (
            now - self._last_publish_time
        ) >= self._min_publish_interval_sec
        should_publish = (is_critical or self._always_publish) and interval_elapsed

        status = {
            "frame_id": self._frame_id,
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "is_critical": is_critical,
            "provider": result.provider_used,
            "inference_latency_ms": result.latency_ms,
            "published": False,
        }

        if not should_publish:
            return status

        anomaly = AnomalyAlert(
            frame_id=self._frame_id,
            zone=self._zone,
            anomaly_score=normalized_score,
            threshold=self._alert_threshold,
            is_critical=is_critical,
            description=description,
        )
        telemetry = StreamTelemetry(
            frame_id=self._frame_id,
            source=self._source_name,
            detections=[],
            anomaly=anomaly,
            processing_latency_ms=result.latency_ms,
        )

        published = self._publisher.publish(telemetry)
        if published:
            self._last_publish_time = now
        status["published"] = published
        return status



# Main loop



def run(args: argparse.Namespace) -> None:
    engine = ONNXInferenceEngine(
        args.onnx_model,
        frame_count=args.frame_count,
        img_size=args.img_size,
        in_channels=1,
        prefer_cuda=not args.no_cuda,
    )

    buffer = SlidingWindowFrameBuffer(
        frame_count=args.frame_count, img_size=args.img_size, channels=1
    )

    low, high = load_calibration_bounds(args.calibration_csv)
    logger.info(
        "Calibration bounds from %s: low=%.6f, high=%.6f",
        args.calibration_csv,
        low,
        high,
    )

    publisher = TelemetryPublisher(
        backend_url=args.backend_url,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

    if publisher.health_check():
        logger.info(
            "Backend at %s is reachable and connected to Redis.", args.backend_url
        )
    else:
        logger.warning(
            "Backend at %s is not reachable (or not connected to Redis) at startup. "
            "Processing will continue; telemetry will be dropped until it recovers.",
            args.backend_url,
        )

    processor = ONNXLiveVADPublisher(
        engine=engine,
        buffer=buffer,
        publisher=publisher,
        calibration_low=low,
        calibration_high=high,
        zone=args.zone,
        alert_threshold=args.alert_threshold,
        source_name=args.source_name or str(args.source),
        min_publish_interval_sec=args.min_publish_interval_sec,
        always_publish=args.always_publish,
    )

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        publisher.close()
        raise RuntimeError(f"Could not open video source: {args.source}")

    logger.info(
        "Reading frames from: %s (publishing to %s/ingest/telemetry, provider=%s)",
        args.source,
        args.backend_url,
        engine.provider,
    )

    frame_idx = 0
    is_live_stream = isinstance(source, int) or str(source).startswith(
        ("rtsp://", "http://", "https://")
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_live_stream:
                    logger.warning("Stream read failed, retrying...")
                    time.sleep(1.0)
                    continue
                logger.info("End of video file reached.")
                break

            frame_idx += 1
            status = processor.process_frame(frame)

            if status is None:
                continue  # still filling the rolling buffer

            if status["published"]:
                logger.info(
                    "[frame %d] score=%.3f critical=%s -> PUBLISHED (%s, %.1fms): %s",
                    status["frame_id"],
                    status["normalized_score"],
                    status["is_critical"],
                    status["provider"],
                    status["inference_latency_ms"],
                    "alert" if status["is_critical"] else "heartbeat",
                )
            elif frame_idx % args.print_every == 0:
                logger.info(
                    "[frame %d] score=%.3f (not published: below threshold or throttled)",
                    status["frame_id"],
                    status["normalized_score"],
                )

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        cap.release()
        publisher.close()
        logger.info("Processed %d frames.", frame_idx)



# CLI



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ONNX-backed live VAD inference, publishing telemetry to the FastAPI backend."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path, webcam index, or RTSP/HTTP URL.",
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        required=True,
        help="Path to the exported .onnx model.",
    )
    parser.add_argument(
        "--calibration_csv",
        type=str,
        required=True,
        help="evaluate.py's per-frame output CSV, used to fix the score scale.",
    )
    parser.add_argument("--backend_url", type=str, default="http://localhost:8000")
    parser.add_argument("--frame_count", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--zone", type=str, default="ZONE_A")
    parser.add_argument("--alert_threshold", type=float, default=0.65)
    parser.add_argument(
        "--min_publish_interval_sec",
        type=float,
        default=2.0,
        help="Minimum seconds between two published telemetry payloads.",
    )
    parser.add_argument(
        "--always_publish",
        action="store_true",
        help="Publish every scored frame (not just threshold-crossing alerts), "
        "for a smoother live chart on the dashboard.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force CPU execution, even if CUDA is available.",
    )
    parser.add_argument("--request_timeout", type=float, default=2.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--source_name", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=30)
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run(cli_args)
