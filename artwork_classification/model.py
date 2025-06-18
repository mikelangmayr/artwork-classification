class ArtworkClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ArtworkClassifier, self).__init__()
        self.base_model = torchvision.models.resnet34(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = True