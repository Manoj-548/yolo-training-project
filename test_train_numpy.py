import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import unittest
from train_numpy import load_data, one_hot, SimpleCNN

class TestTrainNumpy(unittest.TestCase):

    def setUp(self):
        # Create temporary directories and files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = Path(self.temp_dir) / "images"
        self.labels_dir = Path(self.temp_dir) / "labels"
        self.images_dir.mkdir()
        self.labels_dir.mkdir()

        # Create dummy image and label files
        for i in range(3):
            img_path = self.images_dir / f"{i}.jpg"
            label_path = self.labels_dir / f"{i}.txt"
            # Create a dummy grayscale image (28x28)
            img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
            # Save as jpg
            import cv2
            cv2.imwrite(str(img_path), img)
            # Create label file with dummy bounding box
            with open(label_path, 'w') as f:
                f.write(f"{i % 10} 0.5 0.5 0.1 0.1\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_data(self):
        X, y = load_data(self.images_dir, self.labels_dir)
        self.assertEqual(len(X), 3)
        self.assertEqual(len(y), 3)
        self.assertEqual(X.shape[1], 784)  # 28*28
        self.assertTrue(all(isinstance(label, (int, np.integer)) for label in y))

    def test_one_hot(self):
        y = np.array([0, 1, 2])
        oh = one_hot(y, 3)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(oh, expected)

    def test_simple_cnn_init(self):
        model = SimpleCNN(784, 10)
        self.assertEqual(model.input_size, 784)
        self.assertEqual(model.num_classes, 10)
        self.assertEqual(model.conv1_w.shape, (16, 1, 3, 3))
        self.assertEqual(model.conv2_w.shape, (32, 16, 3, 3))
        self.assertEqual(model.fc1_w.shape, (32*7*7, 128))
        self.assertEqual(model.fc2_w.shape, (128, 10))

    def test_simple_cnn_forward(self):
        model = SimpleCNN(784, 10)
        x = np.random.randn(2, 784)
        output = model.forward(x)
        self.assertEqual(output.shape, (2, 10))

    def test_simple_cnn_backward(self):
        model = SimpleCNN(784, 10)
        x = np.random.randn(2, 784)
        y = np.random.randn(2, 10)
        loss = model.backward(x, y)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

if __name__ == '__main__':
    unittest.main()
