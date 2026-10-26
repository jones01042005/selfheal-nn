import asyncio
import unittest
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import api


class BaselineEndpointTests(unittest.TestCase):
    def test_baseline_predicts_without_dataset_evaluation(self):
        image_bytes = (Path(__file__).parent / "test_images" / "OIP.jpeg").read_bytes()
        image = UploadFile(filename="OIP.jpeg", file=BytesIO(image_bytes))

        with (
            patch.object(api, "_cached_baseline_acc", None),
            patch.object(
                api,
                "_get_baseline_acc",
                side_effect=AssertionError("baseline dataset evaluation was called"),
            ),
            patch.object(
                api,
                "_fast_evaluate",
                side_effect=AssertionError("dataset evaluation was called"),
            ),
        ):
            response = asyncio.run(
                asyncio.wait_for(api.run_baseline(image), timeout=5)
            )

        self.assertEqual(set(response), {"accuracy", "prediction", "confidence"})
        self.assertIsNone(response["accuracy"])
        self.assertIn(response["prediction"], api.CLASSES)
        self.assertGreaterEqual(response["confidence"], 0)
        self.assertLessEqual(response["confidence"], 100)


class BaselineStartupTests(unittest.TestCase):
    def setUp(self):
        self.saved_state = {
            name: getattr(api, name)
            for name in (
                "_cached_baseline_acc",
                "_baseline_evaluation_status",
                "_baseline_evaluation_error",
                "_baseline_startup_started",
                "_baseline_startup_thread",
                "_train_loader",
                "_val_loader",
            )
        }
        api._cached_baseline_acc = None
        api._baseline_evaluation_status = "pending"
        api._baseline_evaluation_error = None
        api._baseline_startup_started = False
        api._baseline_startup_thread = None
        api._train_loader = None
        api._val_loader = None

    def tearDown(self):
        thread = api._baseline_startup_thread
        if thread is not None:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "startup evaluation thread leaked")

        for name, value in self.saved_state.items():
            setattr(api, name, value)

    def test_startup_evaluation_is_background_cached_once_and_reported(self):
        evaluation_started = threading.Event()
        release_evaluation = threading.Event()
        second_loader_call = threading.Event()
        call_lock = threading.Lock()
        loader_calls = 0
        evaluation_calls = 0
        second_result = []

        def get_loaders(include_train=True):
            nonlocal loader_calls
            with call_lock:
                loader_calls += 1
                if loader_calls == 2:
                    second_loader_call.set()
            return None, object()

        def fast_evaluate(*args, **kwargs):
            nonlocal evaluation_calls
            evaluation_calls += 1
            evaluation_started.set()
            if not release_evaluation.wait(timeout=5):
                raise TimeoutError("test did not release baseline evaluation")
            return 87.25

        with (
            patch.object(api, "_get_data_loaders", side_effect=get_loaders),
            patch.object(api, "_fast_evaluate", side_effect=fast_evaluate),
            TestClient(api.app) as client,
        ):
            self.assertTrue(evaluation_started.wait(timeout=5))
            api._start_baseline_evaluation()

            running = client.get("/api/health").json()
            self.assertEqual(running["baseline_accuracy_status"], "running")
            self.assertIsNone(running["baseline_accuracy"])
            self.assertIsNone(running["baseline_accuracy_error"])
            self.assertEqual(running["status"], "ok")
            self.assertIn("baseline_checkpoint", running)

            concurrent_result = threading.Thread(
                target=lambda: second_result.append(api._get_baseline_acc())
            )
            concurrent_result.start()
            self.assertTrue(second_loader_call.wait(timeout=5))
            release_evaluation.set()
            concurrent_result.join(timeout=5)
            self.assertFalse(concurrent_result.is_alive())

            startup_thread = api._baseline_startup_thread
            self.assertIsNotNone(startup_thread)
            startup_thread.join(timeout=5)
            self.assertFalse(startup_thread.is_alive())

            completed = client.get("/api/health").json()
            self.assertEqual(completed["baseline_accuracy_status"], "completed")
            self.assertEqual(completed["baseline_accuracy"], 87.25)
            self.assertIsNone(completed["baseline_accuracy_error"])
            self.assertEqual(evaluation_calls, 1)
            self.assertEqual(second_result, [87.25])

    def test_startup_evaluation_failure_is_reported_without_blocking_health(self):
        with (
            patch.object(
                api,
                "_get_data_loaders",
                side_effect=HTTPException(
                    status_code=503,
                    detail="Validation dataset unavailable",
                ),
            ),
            TestClient(api.app) as client,
        ):
            startup_thread = api._baseline_startup_thread
            self.assertIsNotNone(startup_thread)
            startup_thread.join(timeout=5)
            self.assertFalse(startup_thread.is_alive())

            health = client.get("/api/health").json()
            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["baseline_accuracy_status"], "failed")
            self.assertIsNone(health["baseline_accuracy"])
            self.assertEqual(
                health["baseline_accuracy_error"],
                "Validation dataset unavailable",
            )


if __name__ == "__main__":
    unittest.main()
