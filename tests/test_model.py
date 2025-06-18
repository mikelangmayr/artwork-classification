import pytest
import torch
from artwork_classification.model import ArtworkClassifier

def test_model_initialization():
    model = ArtworkClassifier(num_classes=10)
    assert model is not None
    assert isinstance(model, ArtworkClassifier)

def test_forward_pass():
    model = ArtworkClassifier(num_classes=10)
    input_tensor = torch.randn(1, 3, 750, 750)  # Example input tensor
    output = model(input_tensor)
    assert output.shape == (1, 10)  # Check output shape for 10 classes

def test_model_parameters():
    model = ArtworkClassifier(num_classes=10)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0  # Ensure the model has parameters

def test_model_cuda():
    if torch.cuda.is_available():
        model = ArtworkClassifier(num_classes=10).cuda()
        input_tensor = torch.randn(1, 3, 750, 750).cuda()
        output = model(input_tensor)
        assert output.shape == (1, 10)  # Check output shape for 10 classes
    else:
        pytest.skip("CUDA not available, skipping test.")