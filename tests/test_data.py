# File: /artwork-classification/artwork-classification/tests/test_data.py

import unittest
from artwork_classification.data import load_data, split_data

class TestDataFunctions(unittest.TestCase):

    def setUp(self):
        # Setup code to create a mock dataset or use a sample dataset
        self.mock_data = [
            {'image': 'path/to/image1.jpg', 'label': 'Artist1'},
            {'image': 'path/to/image2.jpg', 'label': 'Artist2'},
            {'image': 'path/to/image3.jpg', 'label': 'Artist1'},
        ]

    def test_load_data(self):
        # Test loading data function
        data = load_data('path/to/mock_data')
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_split_data(self):
        # Test splitting data function
        train_data, test_data = split_data(self.mock_data, test_size=0.2)
        self.assertEqual(len(train_data), 2)
        self.assertEqual(len(test_data), 1)

if __name__ == '__main__':
    unittest.main()