#FastAPI backend for telemetry ingestion and real-time streaming.
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import AsyncIterator, List, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from .redis_client import (
    RedisClientError,
    RedisConnectionUnavailableError,
    RedisNotConnectedError,
    RedisTelemetryClient,
)
from .schemas import ErrorResponse, HealthResponse, StreamTelemetry


# Configuration 

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CORS_ALLOWED_ORIGINS: List[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
STREAM_MAX_LENGTH: int = int(os.getenv("TELEMETRY_STREAM_MAX_LENGTH", "2000"))
CATCH_UP_DEFAULT_COUNT: int = int(os.getenv("TELEMETRY_CATCHUP_COUNT", "50"))
REDIS_RECONNECT_MIN_BACKOFF_SEC: float = 1.0
REDIS_RECONNECT_MAX_BACKOFF_SEC: float = 30.0

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("surveillance_backend.app")

redis_client = RedisTelemetryClient(
    redis_url=REDIS_URL,
    max_stream_length=STREAM_MAX_LENGTH,
)



# WebSocket connection manager



class ConnectionManager:
   

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info(
            "WebSocket client connected (%s). Active connections: %d",
            websocket.client,
            len(self._connections),
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info(
            "WebSocket client disconnected. Active connections: %d",
            len(self._connections),
        )

    async def broadcast(self, message: str) -> None:
        async with self._lock:
            targets = list(self._connections)

        if not targets:
            return

        stale: List[WebSocket] = []

        async def _send(connection: WebSocket) -> None:
            try:
                if connection.client_state != WebSocketState.CONNECTED:
                    stale.append(connection)
                    return
                await connection.send_text(message)
            except (WebSocketDisconnect, RuntimeError, ConnectionError) as exc:
                logger.warning("Broadcast failed for a client, marking stale: %s", exc)
                stale.append(connection)

        await asyncio.gather(*(_send(connection) for connection in targets))

        if stale:
            async with self._lock:
                for connection in stale:
                    self._connections.discard(connection)

    @property
    def active_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()
_broadcast_task: "asyncio.Task[None] | None" = None


async def _redis_broadcast_loop() -> None:
    # Pub/Sub listener feeding all WS connections with auto-reconnect.
    backoff = REDIS_RECONNECT_MIN_BACKOFF_SEC

    while True:
        try:
            async for telemetry in redis_client.listen():
                backoff = REDIS_RECONNECT_MIN_BACKOFF_SEC  # reset after any success
                await manager.broadcast(telemetry.model_dump_json())

        except asyncio.CancelledError:
            logger.info("Redis broadcast loop cancelled -- shutting down.")
            raise

        except RedisNotConnectedError:
            logger.warning(
                "Redis not connected yet; retrying broadcast loop in %.1fs.", backoff
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, REDIS_RECONNECT_MAX_BACKOFF_SEC)

        except RedisClientError as exc:
            logger.error(
                "Redis broadcast loop error: %s. Retrying in %.1fs.", exc, backoff
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, REDIS_RECONNECT_MAX_BACKOFF_SEC)

        except Exception:  # noqa: BLE001 -- last-resort guard for a background task
            logger.exception("Unexpected error in Redis broadcast loop.")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, REDIS_RECONNECT_MAX_BACKOFF_SEC)



# App lifespan: connect/disconnect Redis, start/stop broadcast task



@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _broadcast_task

    try:
        await redis_client.connect()
    except RedisConnectionUnavailableError as exc:
       
        logger.error("Redis unavailable at startup: %s", exc)

    _broadcast_task = asyncio.create_task(
        _redis_broadcast_loop(), name="redis-broadcast-loop"
    )
    logger.info("Backend startup complete.")

    yield

    logger.info("Backend shutting down...")
    if _broadcast_task is not None:
        _broadcast_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _broadcast_task

    await redis_client.close()
    logger.info("Backend shutdown complete.")



# App + middleware


app = FastAPI(
    title="AI Surveillance Telemetry Backend",
    description="Decouples the YOLOv8/Conv3D-VAD processing engine from the Streamlit dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RedisClientError)
async def _redis_error_handler(_: object, exc: RedisClientError) -> JSONResponse:
    """Uniform 503 response for any unhandled Redis failure that
    bubbles up out of a route, instead of a raw 500 traceback."""
    logger.error("Unhandled RedisClientError reached the exception handler: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(error="redis_unavailable", detail=str(exc)).model_dump(
            mode="json"
        ),
    )



# REST routes



@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
   
    redis_ok = await redis_client.health_check()
    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        redis_connected=redis_ok,
        active_websocket_connections=manager.active_count,
    )


@app.post(
    "/ingest/telemetry",
    status_code=status.HTTP_202_ACCEPTED,
    responses={503: {"model": ErrorResponse}},
)
async def ingest_telemetry(telemetry: StreamTelemetry) -> dict:
   
    try:
        await redis_client.publish_telemetry(telemetry)
    except RedisNotConnectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Telemetry backend is not connected to Redis: {exc}",
        ) from exc
    except RedisClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to publish telemetry: {exc}",
        ) from exc

    return {"accepted": True, "frame_id": telemetry.frame_id}


@app.get(
    "/telemetry/recent",
    response_model=List[StreamTelemetry],
    response_model_by_alias=False,
    responses={503: {"model": ErrorResponse}},
)
async def get_recent_telemetry(
    count: int = CATCH_UP_DEFAULT_COUNT,
) -> List[StreamTelemetry]:
   
    if count <= 0 or count > 1000:
        raise HTTPException(
            status_code=422,  # Unprocessable Entity/Content, avoids depending on a
            # Starlette status-code constant name that has changed across versions.
            detail="count must be between 1 and 1000.",
        )
    try:
        return await redis_client.read_sliding_window(count=count)
    except RedisNotConnectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Telemetry backend is not connected to Redis: {exc}",
        ) from exc
    except RedisClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to read telemetry stream: {exc}",
        ) from exc



# WebSocket route



@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:


    await manager.connect(websocket)

    try:
        try:
            recent = await redis_client.read_sliding_window(
                count=CATCH_UP_DEFAULT_COUNT
            )
            for telemetry in recent:
                await websocket.send_text(telemetry.model_dump_json())
        except RedisClientError as exc:
            logger.warning("Could not send WebSocket catch-up snapshot: %s", exc)


        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected cleanly.")
    except Exception:  # noqa: BLE001 -- last-resort guard per-connection
        logger.exception("Unexpected error on WebSocket connection.")
    finally:
        await manager.disconnect(websocket)
