import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import time
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=['cornflowerblue', 'hotpink'])})

device = torch.device("cpu")

if __name__ == "__main__":
    train_dir = "Training"
    test_dir = "Testing"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.2251])
    ])

    full_trainset = ImageFolder(root=train_dir, transform=transform)
    testset = ImageFolder(root=test_dir, transform=transform)
    classes = full_trainset.classes
    num_classes = len(classes)

    # Split trainset into train/val (80/20)
    val_size = int(0.2 * len(full_trainset))
    train_size = len(full_trainset) - val_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    learning_rates = [0.01, 0.001, 0.0001]
    epochs_list = [1, 2, 3]
    batch_sizes = [4, 8, 16]

    grid_results = []
    best_macro_f1 = 0
    best_params = None

    print("Starting grid search with macro F1...")

    for lr in learning_rates:
        for num_epochs in epochs_list:
            for batch_size in batch_sizes:

                trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
                valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

                model = resnet50(weights=ResNet50_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model = model.to(device)

                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                criterion = nn.CrossEntropyLoss()

                model.train()
                for epoch in range(num_epochs):
                    for inputs, labels in tqdm(trainloader, desc=f"GS LR={lr}, Epoch={epoch+1}, BS={batch_size}", leave=False):
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                macro_f1 = f1_score(all_labels, all_preds, average='macro')
                grid_results.append({
                    'learning_rate': lr,
                    'epochs': num_epochs,
                    'batch_size': batch_size,
                    'macro_f1': macro_f1
                })

                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_params = {'learning_rate': lr, 'epochs': num_epochs, 'batch_size': batch_size}

    print("Grid search complete. Best params:", best_params)
    pd.DataFrame(grid_results).to_csv("gridsearch_results.csv", index=False)

    # Final model training on full trainset with best params
    final_trainloader = DataLoader(full_trainset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=best_params['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    training_losses = []
    start_time = time.time()
    model.train()
    for epoch in range(best_params['epochs']):
        running_loss = 0.0
        for inputs, labels in tqdm(final_trainloader, desc=f"Final Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(final_trainloader)
        training_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}")
    elapsed_time = time.time() - start_time  
    print(f"CNN Training Time: {elapsed_time:.2f} seconds")

    pd.DataFrame({'epoch': list(range(1, best_params['epochs'] + 1)), 'train_loss': training_losses}).to_csv("training_loss.csv", index=False)

    torch.save(model.state_dict(), "final_cnn_model.pth")
    print("Saved final model weights to 'final_cnn_model.pth'")
