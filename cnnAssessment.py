import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def main():
    device = torch.device("cpu")
    
    # Constants
    train_dir = "Training"
    test_dir = "Testing"
    batch_size = 8
    num_classes = 4  

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.2251])
    ])

    # Test dataset & loader 
    testset = ImageFolder(root=test_dir, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    params = {
        'learning_rate': 0.001,
        'epochs': 3,
        'batch_size': 8
    }

    full_trainset = ImageFolder(root=train_dir, transform=transform)
    final_trainloader = DataLoader(full_trainset, batch_size=params['batch_size'], shuffle=True)

    # Initialize model, optimizer, and loss function before timing
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    #CNN training time for the best parameter 
    start_time = time.time()
    model.train()
    for epoch in range(params['epochs']):
        for inputs, labels in final_trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    elapsed_time = time.time() - start_time
    print(f"CNN Training Time (best params): {elapsed_time:.2f} seconds")

    # Load trained CNN model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("final_cnn_model.pth"))
    model = model.to(device)
    model.eval()

    # Evaluate accuracy, macro F1
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"CNN Test Accuracy: {accuracy:.4f}")
    print(f"CNN Test Macro F1 Score: {macro_f1:.4f}")

    #ROC Curve
    all_probs = []

    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    y_bin = label_binarize(all_labels, classes=np.arange(num_classes))

    # Compute ROC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot micro-average ROC curve (aggregated performance)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr["micro"], tpr["micro"], color='blue', linewidth=1.5, label="CNN")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for CNN")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.box(True) 
    plt.savefig("cnn_roc_curve_simple.png")
    plt.show()

    # Training loss plot
    loss_df = pd.read_csv("training_loss.csv")
    plt.figure(figsize=(8, 5))
    plt.plot(loss_df['epoch'], loss_df['train_loss'], marker='o', color='hotpink', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("CNN Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn_training_loss_curve.png")
    plt.show()

if __name__ == "__main__":
    main()




