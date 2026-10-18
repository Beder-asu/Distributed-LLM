import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from common.metrics import percentiles
from lb.load_balancer import LoadBalancer
from master.scheduler import (
    GPUAwareBalancer,
    LeastConnectionsBalancer,
    LoadAwareBalancer,
)


class QueueStub:
    def __init__(self, depth=0):
        self.depth = depth

    def qsize(self):
        return self.depth


class GPUStub:
    def __init__(self, util_pct=0.0):
        self.util_pct = util_pct


class WorkerStub:
    def __init__(
        self,
        worker_id,
        *,
        inflight=0,
        pending=0,
        util_pct=0.0,
        queue_depth=0,
        concurrency=2,
    ):
        self.id = worker_id
        self.inflight = inflight
        self.pending = pending
        self.concurrency = concurrency
        self.gpu_sim = GPUStub(util_pct)
        self._queue = QueueStub(queue_depth)

    def process(self, request):
        return {"worker_id": self.id, "request": request}


class BalancingStrategyTests(unittest.TestCase):
    def test_round_robin_cycles_in_registration_order(self):
        workers = [WorkerStub(0), WorkerStub(1), WorkerStub(2)]
        balancer = LoadBalancer(workers)

        selected = [balancer.get_next_worker().id for _ in range(7)]

        self.assertEqual(selected, [0, 1, 2, 0, 1, 2, 0])

    def test_least_connections_prefers_lowest_inflight_then_id(self):
        workers = [
            WorkerStub(2, inflight=1),
            WorkerStub(1, inflight=1),
            WorkerStub(0, inflight=3),
        ]

        selected = LeastConnectionsBalancer(workers).get_next_worker()

        self.assertEqual(selected.id, 1)

    def test_load_aware_prefers_lowest_pending_then_id(self):
        workers = [
            WorkerStub(2, pending=2),
            WorkerStub(1, pending=1),
            WorkerStub(0, pending=1),
        ]

        selected = LoadAwareBalancer(workers).get_next_worker()

        self.assertEqual(selected.id, 0)

    def test_gpu_aware_combines_utilization_and_queue_pressure(self):
        workers = [
            WorkerStub(0, util_pct=10, queue_depth=4, concurrency=2),
            WorkerStub(1, util_pct=50, queue_depth=0, concurrency=2),
        ]

        selected = GPUAwareBalancer(workers).get_next_worker()

        # Worker 0 has lower utilization but a saturated queue. Its composite
        # score is 0.46 versus worker 1's 0.30, so worker 1 is the safer pick.
        self.assertEqual(selected.id, 1)


class MetricsTests(unittest.TestCase):
    def test_nearest_rank_percentiles_use_correct_zero_based_index(self):
        self.assertEqual(percentiles([1.0, 2.0], (50, 95)), (1.0, 2.0))
        self.assertEqual(
            percentiles([1.0, 2.0, 3.0, 4.0], (25, 50, 75, 100)),
            (1.0, 2.0, 3.0, 4.0),
        )

    def test_empty_percentiles_are_zero(self):
        self.assertEqual(percentiles([], (50, 95, 99)), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
