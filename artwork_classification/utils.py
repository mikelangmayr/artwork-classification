def visualize_images(images, labels, classes, num_images=4):
    import matplotlib.pyplot as plt
    import numpy as np

    # Function to unnormalize and display images
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # Display a grid of images
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        imshow(images[i])
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.show()

def save_model(model, path):
    """Save the model to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load the model from the specified path."""
    model.load_state_dict(torch.load(path))
    return model

def log_metrics(epoch, loss, accuracy):
    """Log training metrics."""
    print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')