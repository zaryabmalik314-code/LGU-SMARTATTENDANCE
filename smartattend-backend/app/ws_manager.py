"""
Minimal WebSocket broadcast manager for live admin-dashboard updates.

Single-process design (fine for a single Railway instance — no Redis pub/sub
needed at this scale). Admin dashboards connect once after logging in; every
time a check-in or check-out happens, all connected dashboards get pushed
the event instantly instead of needing to refresh.
"""
import asyncio
from typing import Optional
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Called once at app startup so sync endpoints can safely schedule broadcasts."""
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)

    def broadcast_threadsafe(self, message: dict):
        """
        Call this from regular sync (non-async) endpoint functions — FastAPI
        runs sync `def` endpoints in a worker thread, so we can't just
        `await` here. This schedules the broadcast onto the main event loop
        safely from that thread instead.
        """
        if self._loop is None:
            return  # app hasn't finished starting up yet — skip silently
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)


manager = ConnectionManager()
