"""
WebSocket broadcast managers for live dashboard + faculty status updates.

Single-process design (fine for a single Railway instance — no Redis pub/sub
needed at this scale).
"""
import asyncio
from typing import Optional
from fastapi import WebSocket


class ConnectionManager:
    """Broadcasts to all connected admin/HOD dashboards."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
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
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)


class FacultyStatusManager:
    """Per-teacher_id WebSocket connections for pending faculty to receive
    live approval/rejection updates without re-entering their PIN."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, teacher_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections.setdefault(teacher_id, []).append(websocket)

    def disconnect(self, teacher_id: str, websocket: WebSocket):
        conns = self._connections.get(teacher_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            self._connections.pop(teacher_id, None)

    async def notify(self, teacher_id: str, message: dict):
        conns = self._connections.get(teacher_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for d in dead:
            self.disconnect(teacher_id, d)

    def notify_threadsafe(self, teacher_id: str, message: dict):
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.notify(teacher_id, message), self._loop)


manager = ConnectionManager()
faculty_status_manager = FacultyStatusManager()
