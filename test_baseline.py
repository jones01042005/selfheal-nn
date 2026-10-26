import asyncio
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

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


if __name__ == "__main__":
    unittest.main()
