import unittest

try:
    from fastapi.testclient import TestClient
    from python.pcxp_mlops.api import create_app
    from python.pcxp_mlops.predict import PredictionResult
except ModuleNotFoundError:
    TestClient = None
    create_app = None
    PredictionResult = None


class FakeInferenceService:
    def health(self):
        return {"status": "ok", "model_version": "test-version"}

    def model_info(self):
        return {"model_version": "test-version", "metrics": {"recall": 0.91}}

    def predict(self, image_path=None, image_base64=None, threshold=None):
        if image_path == "missing.png":
            raise FileNotFoundError("missing file")
        return PredictionResult(
            predicted_class="PNEUMONIA",
            predicted_index=1,
            probability=0.95,
            threshold=threshold if threshold is not None else 0.87,
            model_version="test-version",
        )


@unittest.skipIf(TestClient is None, "fastapi is not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(FakeInferenceService()))

    def test_health_endpoint_returns_status(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_index_serves_html_interface(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_model_info_endpoint_returns_metadata(self):
        response = self.client.get("/model-info")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], "test-version")

    def test_predict_endpoint_returns_prediction_payload(self):
        response = self.client.post(
            "/predict",
            json={"image_path": "example.png", "threshold": 0.9},
        )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["predicted_class"], "PNEUMONIA")
        self.assertEqual(payload["threshold"], 0.9)
        self.assertIn("request_id", payload)

    def test_predict_endpoint_rejects_multiple_input_sources(self):
        response = self.client.post(
            "/predict",
            json={"image_path": "x.png", "image_base64": "abc"},
        )

        self.assertEqual(response.status_code, 422)

    def test_predict_endpoint_handles_missing_model_input(self):
        response = self.client.post(
            "/predict",
            json={"image_path": "missing.png"},
        )

        self.assertEqual(response.status_code, 400)
