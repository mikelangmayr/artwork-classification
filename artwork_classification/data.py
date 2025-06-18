def load_data(data_dir, transform=None):
    from torchvision import datasets

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset

def split_dataset(dataset, test_size=0.1):
    from torch.utils.data import random_split

    dataset_size = len(dataset)
    test_size = int(test_size * dataset_size)
    train_size = dataset_size - test_size
    return random_split(dataset, [train_size, test_size])

def get_data_loaders(data_dir, transform, batch_size=32, test_size=0.1):
    from torch.utils.data import DataLoader

    dataset = load_data(data_dir, transform)
    train_dataset, test_dataset = split_dataset(dataset, test_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader