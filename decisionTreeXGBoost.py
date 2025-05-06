import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, precision_recall_curve, roc_auc_score, auc, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import pandas as pd 
import matplotlib.pyplot as plt
import time 
import seaborn as sns


def eval_gridsearch(model_name, clf, param_grid, X_train, y_train, X_test, y_test):
    """
    Run GridSearchCV. Time and evaluate best estimator
    """

    # Run GridSearchCV 
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Time the best estimator
    best_estimator = grid.best_estimator_
    start_time = time.time()
    best_estimator.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    preds = best_estimator.predict(X_test)
    
    probs = best_estimator.predict_proba(X_test)

    # Multiclass ROC (micro-average)
    classes = np.unique(y_test)
    y_bin = label_binarize(y_test, classes=classes)

    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    best_params = grid.best_params_

    print(f"\n {model_name} Results")
    print(f"Best Params: {best_params}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Best Estimator Time: {elapsed_time:.2f} seconds")

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_score": f1,
        "best_params": best_params,
        "fpr": fpr,
        "tpr": tpr,
        "time": elapsed_time, 
        "best_estimator": best_estimator if model_name == "XGBoost" else None,
        "grid": grid
    }


def kfold_cv(model, X, y, k=5):
    start_time = time.time()
    kf = KFold(n_splits=k, shuffle=True)

    # ROC AUC and F1 scores for multi-class
    train_auc = np.mean(cross_val_score(model, X, y, cv=kf, scoring='roc_auc_ovr', n_jobs=-1))
    train_f1 = np.mean(cross_val_score(model, X, y, cv=kf, scoring='f1_macro', n_jobs=-1))

    # Get predictions
    y_pred = cross_val_predict(model, X, y, cv=kf, method='predict', n_jobs=-1)
    y_prob = cross_val_predict(model, X, y, cv=kf, method='predict_proba', n_jobs=-1)

    test_auc = roc_auc_score(y, y_prob, multi_class='ovr')
    test_f1 = f1_score(y, y_pred, average='macro')

    # AUPRC (macro) 
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    pr_aucs = []

    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)

    test_auprc = np.mean(pr_aucs)
    train_auprc = test_auprc  # Approximate using same preds (or use cross_val_score if needed)

    elapsed = time.time() - start_time

    return {
        "trainAUC": train_auc,
        "testAUC": test_auc,
        "trainAUPRC": train_auprc,
        "testAUPRC": test_auprc,
        "trainF1": train_f1,
        "testF1": test_f1,
        "timeElapsed": elapsed
    }


def main():
    # Load features
    X_train = np.load('train_features.npy')
    y_train = np.load('train_labels.npy')
    X_test = np.load('test_features.npy')
    y_test = np.load('test_labels.npy')

    results = []

    roc_df = pd.DataFrame()

    # Decision Tree
    dt_params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10]
    }
    dt_clf = DecisionTreeClassifier()
    results.append(eval_gridsearch("Decision Tree", dt_clf, dt_params,
                                X_train, y_train, X_test, y_test))

    #XGBoost
    xgb_params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.05, 0.1]
    }

    xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss')
    results.append(eval_gridsearch("XGBoost", xgb_clf, xgb_params,
                                   X_train, y_train, X_test, y_test))

    # Print summary
    print("\n Summary:")
    for res in results:
        print(f"{res['model']}: Accuracy={res['accuracy']:.4f}, F1={res['f1_score']:.4f}")

    # Plot hyperparameter tuning for Decision Tree
    dt_result = results[0]
    dt_grid = dt_result['grid']
    dt_cv_results = pd.DataFrame(dt_grid.cv_results_)

    plt.figure(figsize=(10, 6))

    for (split, crit) in dt_cv_results[['param_min_samples_split', 'param_criterion']].drop_duplicates().values:
        mask = (dt_cv_results['param_min_samples_split'] == split) & \
            (dt_cv_results['param_criterion'] == crit)
        subset = dt_cv_results[mask]
        plt.plot(subset['param_max_depth'], subset['mean_test_score'], 
                label=f"split={split}, criterion={crit}")

    plt.xlabel("Max Depth")
    plt.ylabel("Mean F1 Score")
    plt.title("Decision Tree: F1 Score vs Max Depth for all Parameter Combos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("decision_tree_all_params.png")
    plt.show()


    # Plot hyperparameter tuning for XGBoost 
    xgb_result = results[1]
    xgb_grid = xgb_result['grid']
    xgb_cv_results = pd.DataFrame(xgb_grid.cv_results_)

    plt.figure(figsize=(10, 6))

    for (lr, est) in xgb_cv_results[['param_learning_rate', 'param_n_estimators']].drop_duplicates().values:
        mask = (xgb_cv_results['param_learning_rate'] == lr) & \
            (xgb_cv_results['param_n_estimators'] == est)
        subset = xgb_cv_results[mask]
        subset = subset.groupby('param_max_depth')['mean_test_score'].mean().reset_index()
        plt.plot(subset['param_max_depth'], subset['mean_test_score'], 
                label=f"lr={lr}, est={est}")

    plt.xlabel("Max Depth")
    plt.ylabel("Mean F1 Score")
    plt.title("XGBoost: F1 Score vs Max Depth for all Param Combos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("xgboost_all_params.png")
    plt.show()

    
    #ROC Comparison Plot 
    for res in results:
        df = pd.DataFrame({
            'fpr': res['fpr'],
            'tpr': res['tpr'],
            'model': res['model']
        })
        roc_df = pd.concat([roc_df, df], ignore_index=True)
    

    color_map = {
        "Decision Tree": "cornflowerblue",
        "XGBoost": "hotpink"
    }

    plt.figure(figsize=(10, 6))
    for model_name in roc_df['model'].unique():
        model_data = roc_df[roc_df['model'] == model_name]
        color = color_map.get(model_name, None)
        plt.plot(model_data['fpr'], model_data['tpr'], label=model_name, color=color)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Decision Tree and XGBoost")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_comparison.png")  
    plt.show()
                     

    for res in results:
        if res['model'] == "XGBoost":
            best_params = res['best_params']
            xgb_best = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='mlogloss')
                
            # Refit on the full training set to visualize training loss only
            xgb_best.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=False
            )
                
            evals_result = xgb_best.evals_result()
                
            # Plot training log loss only
            if 'validation_0' in evals_result:
                plt.figure(figsize=(8, 5))
                plt.plot(evals_result['validation_0']['mlogloss'], label='Training Loss', color='hotpink')
                plt.xlabel('Iteration')
                plt.ylabel('Log Loss')
                plt.title('XGBoost Training Log Loss')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("xgboost_training_loss_only.png")
                plt.show()

    
    print("\nCross-Validation Results:")

    # Cross-validation for Decision Tree
    best_dt_params = results[0]['best_params']
    dt_cv_model = DecisionTreeClassifier(**best_dt_params, random_state=42)
    dt_cv = kfold_cv(dt_cv_model, X_train, y_train, k=5)

    # Cross-validation for XGBoost
    best_xgb_params = results[1]['best_params']
    xgb_cv_model = xgb.XGBClassifier(**best_xgb_params, eval_metric='mlogloss', random_state=42)
    xgb_cv = kfold_cv(xgb_cv_model, X_train, y_train, k=5)

    # Print as DataFrame
    cv_df = pd.DataFrame.from_dict({
        "Decision Tree": dt_cv,
        "XGBoost": xgb_cv
    }, orient='index')
    
    print(cv_df)    

if __name__ == "__main__":
    main()
