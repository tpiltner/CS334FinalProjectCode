import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dir = 'Training'
test_dir = 'Testing'

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  
feature_extractor.eval()

def extract_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = feature_extractor(inputs)  
            outputs = outputs.view(outputs.size(0), -1)  
            features.append(outputs)
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)

print("Extracting train features...")
train_features, train_labels = extract_features(train_loader)
print("Train features extracted!")

print("Extracting test features...")
test_features, test_labels = extract_features(test_loader)
print("Test features extracted!")

# Save as .npy
print("Saving feature arrays...")
np.save('train_features.npy', train_features.numpy())
np.save('train_labels.npy', train_labels.numpy())
np.save('test_features.npy', test_features.numpy())
np.save('test_labels.npy', test_labels.numpy())
print(" Features saved successfully.")


