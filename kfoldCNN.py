import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize


def get_resnet_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def kfold_cv_cnn(model_fn, dataset, num_classes, params, k=5, device='cpu'):
    start_time = time.time()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    fold_f1s = []
    fold_aucs = []
    fold_auprcs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        trainloader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True, num_workers=2)
        valloader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False, num_workers=2)

        model = model_fn(num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(params['epochs']):
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_pred_all.extend(all_preds)
        y_true_all.extend(all_labels)
        y_prob_all.extend(all_probs)

        y_bin = label_binarize(all_labels, classes=range(num_classes))
        probs_np = np.array(all_probs)

        f1 = f1_score(all_labels, all_preds, average='macro')
        auc_score = roc_auc_score(y_bin, probs_np, multi_class='ovr')

        pr_aucs = []
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], probs_np[:, i])
            pr_aucs.append(auc(recall, precision))

        fold_f1s.append(f1)
        fold_aucs.append(auc_score)
        fold_auprcs.append(np.mean(pr_aucs))

    elapsed = time.time() - start_time

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    y_bin_all = label_binarize(y_true_all, classes=range(num_classes))

    test_auc = roc_auc_score(y_bin_all, y_prob_all, multi_class='ovr')
    test_f1 = f1_score(y_true_all, y_pred_all, average='macro')

    test_pr_aucs = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_bin_all[:, i], y_prob_all[:, i])
        test_pr_aucs.append(auc(recall, precision))

    test_auprc = np.mean(test_pr_aucs)
    train_auprc = np.mean(fold_auprcs)

    return {
        "trainAUC": np.mean(fold_aucs),
        "testAUC": test_auc,
        "trainAUPRC": train_auprc,
        "testAUPRC": test_auprc,
        "trainF1": np.mean(fold_f1s),
        "testF1": test_f1,
        "timeElapsed": elapsed
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.2251])
    ])

    train_dir = "Training"
    trainset = ImageFolder(root=train_dir, transform=transform)
    num_classes = len(trainset.classes)

    params = {
        'learning_rate': 0.001,
        'epochs': 3,
        'batch_size': 8
    }

    results = kfold_cv_cnn(get_resnet_model, trainset, num_classes, params, k=5, device=device)
    print("\n K-Fold CNN Results ")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


