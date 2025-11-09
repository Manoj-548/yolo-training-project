import unittest
import subprocess
import sys
import os
from pathlib import Path

class TestMainScript(unittest.TestCase):
    def setUp(self):
        self.script_path = Path(__file__).parent / "main.py"
        self.valid_weights = "yolov8n.pt"
        self.invalid_weights = "invalid_weights.pt"
        self.valid_source = Path("bottle_char_result.jpg")
        self.invalid_source = Path("non_existent.jpg")
        self.data_yaml = Path("data.yaml")
        # Create a minimal valid data.yaml for testing
        if not self.data_yaml.exists():
            with open(self.data_yaml, "w") as f:
                f.write("names:\n  0: test_class\n")

    def run_script(self, args):
        cmd = [sys.executable, str(self.script_path)] + args
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.stdout is None:
            result.stdout = ''
        if result.stderr is None:
            result.stderr = ''
        return result

    def test_dependency_check(self):
        # Run script with no args to trigger dependency check
        result = self.run_script([])
        self.assertNotIn("Missing dependencies", result.stdout + result.stderr)

    def test_invalid_weights(self):
        result = self.run_script(["--weights", self.invalid_weights])
        self.assertIn("Invalid weights file", result.stdout + result.stderr)

    def test_missing_data_yaml(self):
        result = self.run_script(["--data", "non_existent.yaml", "--weights", self.valid_weights, "--skip-train"])
        self.assertIn("Data file not found", result.stdout + result.stderr)

    def test_invalid_data_yaml(self):
        # Create invalid yaml
        with open("invalid_data.yaml", "w") as f:
            f.write("invalid: [")
        result = self.run_script(["--data", "invalid_data.yaml", "--weights", self.valid_weights, "--skip-train"])
        self.assertIn("Failed to read YAML", result.stdout + result.stderr)
        os.remove("invalid_data.yaml")

    def test_training_skipped(self):
        result = self.run_script(["--weights", self.valid_weights, "--skip-train"])
        self.assertIn("No source provided", result.stdout + result.stderr)

    def test_inference_with_valid_source(self):
        result = self.run_script(["--weights", self.valid_weights, "--source", str(self.valid_source), "--skip-train"])
        self.assertIn("Inference on", result.stdout + result.stderr)

    def test_inference_with_invalid_source(self):
        result = self.run_script(["--weights", self.valid_weights, "--source", str(self.invalid_source), "--skip-train"])
        self.assertIn("Source path not found", result.stdout + result.stderr)

    def tearDown(self):
        # Clean up created files
        if self.data_yaml.exists():
            self.data_yaml.unlink()

if __name__ == "__main__":
    unittest.main()
