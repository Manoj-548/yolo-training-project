import unittest
import requests
import os
import subprocess
import time
from pathlib import Path

class TestUnifiedAPI(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8000"

    def setUp(self):
        # Start the server in a subprocess
        self.server_process = subprocess.Popen(
            ["python", "unified_api.py"],
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # Wait for server to start

    def tearDown(self):
        self.server_process.terminate()
        self.server_process.wait()

    def test_timeline_endpoint(self):
        response = requests.get(f"{self.BASE_URL}/timeline")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("timestamp", data)
        self.assertIn("repos", data)

    def test_run_endpoint_valid_repo(self):
        payload = {"input": "test", "repo": "ml_deploy"}
        response = requests.post(f"{self.BASE_URL}/run", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("timestamp", data)
        self.assertIn("output", data)

    def test_run_endpoint_invalid_repo(self):
        payload = {"input": "test", "repo": "invalid"}
        response = requests.post(f"{self.BASE_URL}/run", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("error", data)

    def test_update_endpoint_valid_repo(self):
        response = requests.get(f"{self.BASE_URL}/update/ml_deploy")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("timestamp", data)
        self.assertIn("output", data)

    def test_update_endpoint_invalid_repo(self):
        response = requests.get(f"{self.BASE_URL}/update/invalid")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("error", data)

    def test_openrouter_endpoint_mock(self):
        # Set mock API key for testing
        os.environ["OPENROUTER_API_KEY"] = "mock"
        payload = {"input": "Hello, world!", "model": "custom/blackbox-base"}
        response = requests.post(f"{self.BASE_URL}/openrouter", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertIn("model", data)
        self.assertEqual(data["model"], "custom/blackbox-base")

    def test_openrouter_endpoint_no_api_key(self):
        # Remove API key
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]
        payload = {"input": "Hello, world!", "model": "custom/blackbox-base"}
        response = requests.post(f"{self.BASE_URL}/openrouter", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

if __name__ == "__main__":
    unittest.main()
