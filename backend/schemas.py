"""Shared Pydantic schemas for the ML engine, backend API, and dashboard.
Field aliases match legacy CSV column names so rows can be validated directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utcnow() -> datetime:
    
    return datetime.now(timezone.utc)


class EventType(str, Enum):
   

    INTRUSION = "Intrusion"
    LOITERING = "Loitering"
    FALL = "Fall Detected"
    OCCUPANCY_ENTRY = "ENTRY"
    OCCUPANCY_EXIT = "EXIT"
    BEHAVIOR = "Behavior"
    ABSTRACT_ANOMALY = "Abstract Anomaly"


class BoundingBox(BaseModel):
   

    model_config = ConfigDict(frozen=True, extra="forbid")

    x1: float = Field(..., description="Left edge, in pixels.")
    y1: float = Field(..., description="Top edge, in pixels.")
    x2: float = Field(..., description="Right edge, in pixels.")
    y2: float = Field(..., description="Bottom edge, in pixels.")

    @model_validator(mode="after")
    def _validate_positive_area(self) -> "BoundingBox":
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be greater than x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be greater than y1 ({self.y1})")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class DetectionEvent(BaseModel):
    

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    frame_id: int = Field(
        ..., ge=0, description="Monotonic frame counter from the video loop."
    )
    person_id: Optional[int] = Field(
        default=None,
        ge=0,
        alias="Person_ID",
        description="YOLOv8 tracker ID, if applicable.",
    )
    event_type: EventType = Field(..., alias="Event")
    zone: Optional[str] = Field(default=None, alias="Zone", min_length=1, max_length=64)
    bbox: Optional[BoundingBox] = Field(
        default=None, description="Absolute pixel bounding box."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detector confidence score."
    )
    timestamp: datetime = Field(default_factory=_utcnow, alias="Timestamp")

    @field_validator("zone")
    @classmethod
    def _normalize_zone(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("zone, if provided, must not be blank")
        return normalized


class AnomalyAlert(BaseModel):
    

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    frame_id: int = Field(..., ge=0)
    zone: str = Field(default="ZONE_A", alias="Zone", min_length=1, max_length=64)
    anomaly_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized VAD reconstruction-error score."
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The calibrated safety threshold this was compared against.",
    )
    is_critical: bool = Field(
        ..., description="anomaly_score >= threshold at publish time."
    )
    description: str = Field(..., alias="Description", min_length=1, max_length=512)
    timestamp: datetime = Field(default_factory=_utcnow, alias="Timestamp")

    @model_validator(mode="after")
    def _validate_criticality_consistency(self) -> "AnomalyAlert":
        expected_critical = self.anomaly_score >= self.threshold
        if self.is_critical != expected_critical:
            raise ValueError(
                f"is_critical={self.is_critical} is inconsistent with "
                f"anomaly_score={self.anomaly_score} vs threshold={self.threshold}"
            )
        return self


class StreamTelemetry(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    frame_id: int = Field(..., ge=0)
    source: str = Field(default="camera_1", min_length=1, max_length=128)
    detections: List[DetectionEvent] = Field(default_factory=list)
    anomaly: Optional[AnomalyAlert] = Field(default=None)
    processing_latency_ms: float = Field(
        ..., ge=0.0, description="Wall-clock time to process this frame, end to end."
    )
    published_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _validate_frame_id_consistency(self) -> "StreamTelemetry":
        mismatched = [
            d.frame_id for d in self.detections if d.frame_id != self.frame_id
        ]
        if mismatched:
            raise ValueError(
                f"detections contain frame_id(s) {sorted(set(mismatched))} "
                f"that do not match the telemetry envelope's frame_id={self.frame_id}"
            )
        if self.anomaly is not None and self.anomaly.frame_id != self.frame_id:
            raise ValueError(
                f"anomaly.frame_id={self.anomaly.frame_id} does not match "
                f"the telemetry envelope's frame_id={self.frame_id}"
            )
        return self


class HealthResponse(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., pattern="^(ok|degraded)$")
    redis_connected: bool
    active_websocket_connections: int = Field(..., ge=0)
    checked_at: datetime = Field(default_factory=_utcnow)


class ErrorResponse(BaseModel):
   

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: Optional[str] = None
