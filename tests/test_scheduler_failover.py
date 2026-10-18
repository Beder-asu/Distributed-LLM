import asyncio
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from common.models import Request
from master.scheduler import Scheduler


class AsyncWorkerStub:
    def __init__(self, worker_id, *, error=None, delay=0.0):
        self.id = worker_id
        self.error = error
        self.delay = delay
        self.calls = 0
        self.inflight = 0
        self.pending = 0
        self.processed_count = 0
        self.total_latency = 0.0
        self.last_heartbeat = time.time()
        self.recent = []
        self.queue_wait = []
        self.end_to_end = []
        self.concurrency = 1
        self._queue = None
        self.gpu_sim = SimpleNamespace(util_pct=0.0)

    async def submit(self, request):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return {
            "id": request.id,
            "worker_id": self.id,
            "result": "ok",
            "latency": 0.001,
        }


class SchedulerFailoverTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_request_is_reassigned_to_an_untried_worker(self):
        failed = AsyncWorkerStub(0, error=RuntimeError("simulated worker failure"))
        healthy = AsyncWorkerStub(1)
        scheduler = Scheduler([failed, healthy], strategy="round_robin")

        result = await scheduler.handle_request_async(Request(7, "test"))

        self.assertEqual(result["worker_id"], 1)
        self.assertEqual((failed.calls, healthy.calls), (1, 1))
        self.assertEqual(scheduler.metrics.retries, 1)
        self.assertEqual(scheduler.metrics.failures, 1)
        self.assertEqual((failed.pending, healthy.pending), (0, 0))

    async def test_timeout_is_recorded_before_successful_reassignment(self):
        slow = AsyncWorkerStub(0, delay=0.05)
        healthy = AsyncWorkerStub(1)
        scheduler = Scheduler([slow, healthy], strategy="round_robin")
        scheduler.REQUEST_TIMEOUT = 0.005

        result = await scheduler.handle_request_async(Request(8, "test"))

        self.assertEqual(result["worker_id"], 1)
        self.assertEqual(scheduler.metrics.timeouts, 1)
        self.assertEqual(scheduler.metrics.retries, 1)
        self.assertEqual((slow.pending, healthy.pending), (0, 0))

    async def test_exhausted_workers_surface_the_last_error(self):
        workers = [
            AsyncWorkerStub(0, error=ValueError("first")),
            AsyncWorkerStub(1, error=RuntimeError("second")),
        ]
        scheduler = Scheduler(workers, strategy="round_robin")

        with self.assertRaisesRegex(RuntimeError, "failed after 2 attempt"):
            await scheduler.handle_request_async(Request(9, "test"))

        self.assertEqual(scheduler.metrics.failures, 2)
        self.assertEqual(scheduler.metrics.retries, 1)


if __name__ == "__main__":
    unittest.main()
