import pytest
import torch
from torchvision import transforms
from artwork_classification.transforms import AddGaussianNoise

def test_add_gaussian_noise():
    # Create a sample tensor image
    sample_image = torch.rand(3, 256, 256)  # Random image with 3 channels (RGB)
    
    # Initialize the AddGaussianNoise transform
    noise_transform = AddGaussianNoise(mean=0.0, std=0.1)
    
    # Apply the transform
    noisy_image = noise_transform(sample_image)
    
    # Check if the output is the same shape as the input
    assert noisy_image.shape == sample_image.shape, "Output shape should match input shape"
    
    # Check if the noise is added (the output should not be exactly the same as the input)
    assert not torch.equal(sample_image, noisy_image), "Noisy image should differ from the original image"

def test_transforms_composition():
    # Create a sample tensor image
    sample_image = torch.rand(3, 256, 256)
    
    # Define a series of transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.1)
    ])
    
    # Apply the transformations
    transformed_image = transform(sample_image)
    
    # Check if the output shape is correct after resizing
    assert transformed_image.shape == (3, 128, 128), "Transformed image should be resized to (3, 128, 128)"