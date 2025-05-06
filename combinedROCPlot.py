import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

device = torch.device("cpu")
test_dir = "Testing"
num_classes = 4
# best CNN batch size
batch_size = 8  

# Load test set for CNN
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.2251])
])
testset = ImageFolder(root=test_dir, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

cnn_model = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, num_classes)
cnn_model.load_state_dict(torch.load("final_cnn_model.pth", map_location=device))
cnn_model = cnn_model.to(device)
cnn_model.eval()

cnn_probs = []
cnn_labels = []
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        outputs = cnn_model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        cnn_probs.extend(probs.cpu().numpy())
        cnn_labels.extend(labels.cpu().numpy())
cnn_probs = np.array(cnn_probs)
cnn_labels = np.array(cnn_labels)
cnn_y_bin = label_binarize(cnn_labels, classes=np.arange(num_classes))
fpr_cnn, tpr_cnn, _ = roc_curve(cnn_y_bin.ravel(), cnn_probs.ravel())

X_train = np.load("train_features.npy")
y_train = np.load("train_labels.npy")
X_test = np.load("test_features.npy")
y_test = np.load("test_labels.npy")
y_bin = label_binarize(y_test, classes=np.unique(y_test))

# Decision tree (best parameters)
dt_best = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=2)
dt_best.fit(X_train, y_train)
dt_probs = dt_best.predict_proba(X_test)
fpr_dt, tpr_dt, _ = roc_curve(y_bin.ravel(), dt_probs.ravel())

# XGBoost (best parameters)
xgb_best = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='mlogloss')
xgb_best.fit(X_train, y_train)
xgb_probs = xgb_best.predict_proba(X_test)
fpr_xgb, tpr_xgb, _ = roc_curve(y_bin.ravel(), xgb_probs.ravel())

# Plot the combined ROC curves (micro-averaged)
plt.figure(figsize=(10, 6))
plt.plot(fpr_dt, tpr_dt, label="Decision Tree", color="cornflowerblue", linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost", color="hotpink", linewidth=2)
plt.plot(fpr_cnn, tpr_cnn, label="CNN", color="seagreen", linewidth=2)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Brain Tumor Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_roc.png")
plt.show()
