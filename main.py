import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Identifying columns
column_names = ['ip_mean', 'ip_std', 'ip_kurt', 'ip_skew', 'dm_mean', 'dm_std', 'dm_kurt', 'dm_skew', 'class']

df = pd.read_csv("HTRU_2.csv", header = None, names = column_names)

X = df.drop(columns="class")
Y = df["class"]


# Training and testing
# We have to stratify because our data isn't balanced
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=13
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # fit only on training
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=13, n_jobs=-1
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=13
    ),
}

preds = {}
for name, model in models.items():
    model.fit(X_train, Y_train)
    preds[name] = model.predict(X_test)
    print(f"=== {name} ===")
    print(confusion_matrix(Y_test, preds[name]))
    print(classification_report(Y_test, preds[name]))

n_models = len(models)
fig = plt.figure(figsize=(6 * n_models, 11))
grid = fig.add_gridspec(2, 2 * n_models)
cm_axes = [fig.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(n_models)]
score_ax = fig.add_subplot(grid[1, :n_models])
pr_ax = fig.add_subplot(grid[1, n_models:])

# 1. Confusion matrix for each model
for ax, (name, pred) in zip(cm_axes, preds.items()):
    ConfusionMatrixDisplay.from_predictions(
        Y_test, pred,
        display_labels=["Not pulsar", "Pulsar"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"{name}: confusion matrix")

# 2. Pulsar-class precision / recall / F1 side by side
metrics = ["Precision", "Recall", "F1"]
x = np.arange(len(metrics))
width = 0.8 / n_models
for i, (name, pred) in enumerate(preds.items()):
    precision, recall, f1, _ = precision_recall_fscore_support(Y_test, pred)
    scores = [precision[1], recall[1], f1[1]]
    offset = (i - (n_models - 1) / 2) * width
    bars = score_ax.bar(x + offset, scores, width, label=name)
    score_ax.bar_label(bars, fmt="%.2f", fontsize=8)
score_ax.set_xticks(x, metrics)
score_ax.set_ylim(0, 1.1)
score_ax.set_title("Pulsar class scores")
score_ax.legend(loc="lower right")

# 3. Precision-recall curves on the same axes
for name, model in models.items():
    PrecisionRecallDisplay.from_estimator(model, X_test, Y_test, name=name, ax=pr_ax)
pr_ax.set_title("Precision-recall curve")

plt.tight_layout()
plt.show()


