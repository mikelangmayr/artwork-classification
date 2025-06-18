# Contents of /artwork-classification/artwork-classification/tests/test_train.py

import unittest
from artwork_classification.train import train_model
from artwork_classification.model import MyModel
from artwork_classification.data import get_data_loaders

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        self.model = MyModel()
        self.train_loader, self.test_loader = get_data_loaders(batch_size=4)

    def test_train_model(self):
        # Test if the model can be trained without errors
        try:
            train_model(self.model, self.train_loader, num_epochs=1)
        except Exception as e:
            self.fail(f"train_model raised an exception: {e}")

    def test_model_output_shape(self):
        # Test if the model output shape is correct
        inputs, _ = next(iter(self.train_loader))
        outputs = self.model(inputs)
        self.assertEqual(outputs.shape[1], 50)  # Assuming 50 classes

if __name__ == '__main__':
    unittest.main()