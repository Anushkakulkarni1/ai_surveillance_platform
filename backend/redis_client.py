"""Async Redis client for telemetry storage and pub/sub streaming.
Uses a bounded Redis Stream for history catch-up queries and Pub/Sub for live updates.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, List, Optional

import redis.asyncio as redis
from redis.asyncio.client import PubSub
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError, TimeoutError as RedisTimeoutError

from .schemas import StreamTelemetry

logger = logging.getLogger("surveillance_backend.redis_client")


class RedisClientError(Exception):
    """Base class for all errors raised by RedisTelemetryClient."""


class RedisConnectionUnavailableError(RedisClientError):
    """Raised when a connection to Redis could not be established."""


class RedisNotConnectedError(RedisClientError):
    """Raised when a method requiring an active connection is called
    before connect() has succeeded."""


class RedisTelemetryClient:
    

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        stream_key: str = "telemetry:stream",
        channel: str = "telemetry:live",
        max_stream_length: int = 2000,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
    ) -> None:
        if max_stream_length <= 0:
            raise ValueError(
                f"max_stream_length must be positive, got {max_stream_length}"
            )
        if socket_timeout <= 0 or socket_connect_timeout <= 0:
            raise ValueError(
                "socket_timeout and socket_connect_timeout must be positive"
            )

        self._redis_url = redis_url
        self._stream_key = stream_key
        self._channel = channel
        self._max_stream_length = max_stream_length
        self._socket_timeout = socket_timeout
        self._socket_connect_timeout = socket_connect_timeout

        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._connect_lock = asyncio.Lock()

   
    # Lifecycle
   

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        
        async with self._connect_lock:
            if self._client is not None:
                return
            try:
                self._pool = redis.ConnectionPool.from_url(
                    self._redis_url,
                    decode_responses=True,
                    socket_timeout=self._socket_timeout,
                    socket_connect_timeout=self._socket_connect_timeout,
                )
                client = redis.Redis(connection_pool=self._pool)
                await client.ping()
                self._client = client
                logger.info("Connected to Redis at %s", self._redis_url)
            except (RedisConnectionError, RedisTimeoutError, OSError) as exc:
                self._client = None
                if self._pool is not None:
                    await self._pool.disconnect()
                    self._pool = None
                raise RedisConnectionUnavailableError(
                    f"Could not connect to Redis at {self._redis_url}: {exc}"
                ) from exc

    async def close(self) -> None:
       
        async with self._connect_lock:
            if self._client is not None:
                try:
                    await self._client.aclose()
                except RedisError as exc:
                    logger.warning("Error while closing Redis client: %s", exc)
                self._client = None
            if self._pool is not None:
                try:
                    await self._pool.disconnect()
                except RedisError as exc:
                    logger.warning("Error while disconnecting Redis pool: %s", exc)
                self._pool = None
            logger.info("Redis connection closed.")

    def _require_client(self) -> redis.Redis:
        if self._client is None:
            raise RedisNotConnectedError(
                "RedisTelemetryClient is not connected -- call connect() first."
            )
        return self._client

    async def health_check(self) -> bool:
        
        if self._client is None:
            return False
        try:
            pong = await self._client.ping()
            return bool(pong)
        except RedisError as exc:
            logger.warning("Redis health check failed: %s", exc)
            return False

   
   
   

    async def publish_telemetry(self, telemetry: StreamTelemetry) -> None:
        
        client = self._require_client()
        payload = telemetry.model_dump_json()

        try:
            await client.xadd(
                self._stream_key,
                {"payload": payload},
                maxlen=self._max_stream_length,
                approximate=True,
            )
        except RedisError as exc:
            logger.error(
                "Failed to XADD telemetry frame_id=%s: %s", telemetry.frame_id, exc
            )
            raise RedisClientError(
                f"Failed to push telemetry to stream: {exc}"
            ) from exc

        try:
            await client.publish(self._channel, payload)
        except RedisError as exc:
           
            logger.error(
                "Telemetry frame_id=%s was buffered but broadcast failed: %s",
                telemetry.frame_id,
                exc,
            )
            raise RedisClientError(
                f"Failed to publish telemetry to channel: {exc}"
            ) from exc

   
    # Consumer side: sliding-window buffer (REST catch-up / polling)
   

    async def read_sliding_window(self, count: int = 100) -> List[StreamTelemetry]:
        
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        client = self._require_client()

        try:
            
            entries = await client.xrevrange(self._stream_key, count=count)
        except RedisError as exc:
            raise RedisClientError(f"Failed to read telemetry stream: {exc}") from exc

        results: List[StreamTelemetry] = []
        for entry_id, fields in entries:
            raw_payload = fields.get("payload")
            if raw_payload is None:
                logger.warning(
                    "Stream entry %s missing 'payload' field, skipping.", entry_id
                )
                continue
            try:
                results.append(StreamTelemetry.model_validate_json(raw_payload))
            except ValueError as exc:
                logger.warning("Skipping malformed stream entry %s: %s", entry_id, exc)
                continue

        results.reverse()  
        return results

   
    # Consumer side: real-time Pub/Sub fan-out
   

    async def _subscribe(self) -> PubSub:
        client = self._require_client()
        pubsub = client.pubsub()
        try:
            await pubsub.subscribe(self._channel)
        except RedisError as exc:
            await pubsub.aclose()
            raise RedisClientError(
                f"Failed to subscribe to {self._channel}: {exc}"
            ) from exc
        return pubsub

    async def listen(self) -> AsyncIterator[StreamTelemetry]:
        """Yield live telemetry payloads from the Pub/Sub channel."""
        pubsub = await self._subscribe()
        poll_interval_sec = 1.0

        try:
            while True:
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=poll_interval_sec
                    )
                except RedisError as exc:
                    raise RedisClientError(
                        f"Pub/Sub listen loop failed: {exc}"
                    ) from exc

                if message is None:
                    # Idle poll timeout—nothing published, keep waiting
                    await asyncio.sleep(0)
                    continue

                if message.get("type") != "message":
                    # "subscribe"/"unsubscribe" confirmation messages,
                    # not actual telemetry then ignore.
                    continue

                raw_payload = message.get("data")
                if not raw_payload:
                    continue

                try:
                    yield StreamTelemetry.model_validate_json(raw_payload)
                except ValueError as exc:
                    logger.warning("Skipping malformed pub/sub message: %s", exc)
                    continue
        finally:
            try:
                await pubsub.unsubscribe(self._channel)
            except RedisError as exc:
                logger.warning("Error unsubscribing from %s: %s", self._channel, exc)
            await pubsub.aclose()
