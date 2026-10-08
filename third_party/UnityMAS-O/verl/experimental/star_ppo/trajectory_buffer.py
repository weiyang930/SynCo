import time
import threading
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from verl import DataProto


@dataclass
class TrajectoryEntry:
    traj_id: str
    model_id: str
    query_id: str
    agent_id: str
    fat_data: DataProto
    reward: Optional[torch.Tensor] = None
    done: bool = False
    created_at: float = field(default_factory=time.time)


class TrajectoryBuffer:
    """In-worker trajectory buffer for fat-data residency."""

    def __init__(self, max_items: int, ttl_seconds: int, dropped_query_ttl_seconds: int | None = None):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        if dropped_query_ttl_seconds is None:
            if ttl_seconds > 0:
                dropped_query_ttl_seconds = min(ttl_seconds, 120)
            else:
                dropped_query_ttl_seconds = 120
        self.dropped_query_ttl_seconds = int(max(1, dropped_query_ttl_seconds))
        self.entries: OrderedDict[str, TrajectoryEntry] = OrderedDict()
        self.ready_queue: deque[str] = deque()
        self.dropped_queries: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.RLock()

    def _prune_ready_queue(self, removed_ids: set[str]) -> None:
        if not removed_ids or len(self.ready_queue) == 0:
            return
        self.ready_queue = deque([traj_id for traj_id in self.ready_queue if traj_id not in removed_ids])

    def _evict_expired(self) -> None:
        with self._lock:
            if self.ttl_seconds <= 0:
                return
            now = time.time()
            # Iterate over a snapshot to avoid runtime errors when entries are updated concurrently.
            expired = [k for k, v in list(self.entries.items()) if now - v.created_at > self.ttl_seconds]
            if not expired:
                return
            removed_ids = set(expired)
            for key in expired:
                self.entries.pop(key, None)
            self._prune_ready_queue(removed_ids)

    def _evict_expired_dropped_queries(self) -> None:
        with self._lock:
            if self.dropped_query_ttl_seconds <= 0:
                return
            now = time.time()
            expired = [
                q
                for q, ts in list(self.dropped_queries.items())
                if now - ts > self.dropped_query_ttl_seconds
            ]
            for query_id in expired:
                self.dropped_queries.pop(query_id, None)

    def _purge_query_nolock(self, query_id: str) -> int:
        if not query_id:
            return 0
        removed_ids: set[str] = set()
        for traj_id, entry in list(self.entries.items()):
            if entry.query_id == query_id:
                self.entries.pop(traj_id, None)
                removed_ids.add(traj_id)
        self._prune_ready_queue(removed_ids)
        return len(removed_ids)

    def _evict_overflow(self) -> None:
        with self._lock:
            removed_ids: set[str] = set()
            while len(self.entries) > self.max_items:
                oldest_key = next(iter(self.entries.keys()))
                self.entries.pop(oldest_key, None)
                removed_ids.add(oldest_key)
            self._prune_ready_queue(removed_ids)

    def put(self, entry: TrajectoryEntry) -> None:
        with self._lock:
            self._evict_expired()
            self._evict_expired_dropped_queries()
            if entry.query_id in self.dropped_queries:
                return
            self.entries[entry.traj_id] = entry
            self._evict_overflow()

    def mark_query_dropped(self, query_id: str) -> int:
        with self._lock:
            self._evict_expired()
            self._evict_expired_dropped_queries()
            if not query_id:
                return 0
            self.dropped_queries[query_id] = time.time()
            return self._purge_query_nolock(query_id)

    def mark_queries_dropped(self, query_ids: list[str]) -> int:
        with self._lock:
            self._evict_expired()
            self._evict_expired_dropped_queries()
            purged = 0
            for query_id in query_ids:
                q = str(query_id or "")
                if not q:
                    continue
                self.dropped_queries[q] = time.time()
                purged += self._purge_query_nolock(q)
            return purged

    def commit_reward(self, traj_id: str, reward: torch.Tensor, done: bool) -> bool:
        with self._lock:
            self._evict_expired()
            self._evict_expired_dropped_queries()
            entry = self.entries.get(traj_id)
            if entry is None:
                return False

            if entry.query_id in self.dropped_queries:
                self.entries.pop(traj_id, None)
                return False

            was_done = entry.done
            entry.reward = reward.detach().cpu()
            entry.done = bool(done)
            if entry.done and not was_done:
                self.ready_queue.append(traj_id)
            return True

    def pop_ready(self, max_items: int | None = None, *, shuffle: bool = False) -> list[TrajectoryEntry]:
        with self._lock:
            self._evict_expired()
            self._evict_expired_dropped_queries()
            out: list[TrajectoryEntry] = []
            limit = max_items if max_items is not None and max_items > 0 else len(self.ready_queue)
            if shuffle and len(self.ready_queue) > 1:
                ready_ids = list(self.ready_queue)
                random.shuffle(ready_ids)
                self.ready_queue = deque(ready_ids)
            for _ in range(min(limit, len(self.ready_queue))):
                traj_id = self.ready_queue.popleft()
                entry = self.entries.pop(traj_id, None)
                if entry is not None:
                    out.append(entry)
            return out

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._evict_expired()
            self._evict_expired_dropped_queries()
            return {
                "buffer/total": len(self.entries),
                "buffer/ready": len(self.ready_queue),
                "buffer/dropped_queries": len(self.dropped_queries),
            }
