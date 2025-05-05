import pandas as pd
import matplotlib.pyplot as plt

# Load grid search CSV file
df = pd.read_csv("gridsearch_results.csv")  

# Create a label combining learning rate and epochs
df['label'] = df.apply(lambda row: f"lr={row['learning_rate']}, ep={row['epochs']}", axis=1)

# Plot Macro F1 (%) vs Batch Size
plt.figure(figsize=(10, 6))
for label, group in df.groupby("label"):
    group_sorted = group.sort_values("batch_size")
    # Multiply macro_f1 by 100 to convert to percentage
    plt.plot(group_sorted["batch_size"], group_sorted["macro_f1"] * 100, marker="o", label=label)

plt.title("CNN Macro F1 vs Batch Size for All Hyperparameter Combos")
plt.xlabel("Batch Size")
plt.ylabel("Macro F1 Score (%)")  

# Place legend 
plt.legend(title="Learning Rate & Epochs", loc='lower right', fontsize=10, title_fontsize=11)

plt.grid(True)
plt.tight_layout()
plt.show()




