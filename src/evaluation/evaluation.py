import os
import sys
import itertools
import h5py
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix


sys.path.append("src")
from model_training.architectures import SliceLayer


def precision_recall_plot(data_dict):
    plt.figure(figsize=(8, 8))

    for model_name, data in data_dict.items():
        labels = data["labels"]
        preds = data["preds"]
        precisions, recalls, _ = precision_recall_curve(labels, preds)
        pr_ap = auc(recalls, precisions)
        plt.plot(recalls, precisions, label=f"{model_name} (AP = {pr_ap:.4f})", linewidth=2)

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(os.path.join("plots", "precision_recall_curves.png"), dpi=300)


def auc_roc_plot(data_dict):
    plt.figure(figsize=(8, 8))

    for model_name, data in data_dict.items():
        labels = data["labels"]
        preds = data["preds"]
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.4f})", linewidth=2)

    plt.title("ROC Curves")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(os.path.join("plots", "roc_curves.png"), dpi=300)


def confusion_matrix_plot(data_dict):
    for model_name, data in data_dict.items():
        labels = data["labels"]
        preds = np.round(data["preds"])
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.colorbar()
        tick_marks = np.arange(len(["Actual Negative", "Actual Positive"]))
        plt.xticks(tick_marks, ["Predicted Negative", "Predicted Positive"], rotation=45)
        plt.yticks(tick_marks, ["Actual Negative", "Actual Positive"])

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(f"plots/confusion_matrix_{model_name}.png", dpi=300)
        plt.show()


def load_data(testing_data_path):
    with h5py.File(testing_data_path, "r") as file:
        dataset_names = list(file.keys())
        data = []
        labels = []
        for dataset in dataset_names:
            data.append(file[dataset][:])
            labels.append(file[dataset].attrs["label"])
    return np.array(data), np.array(labels)[:, np.newaxis]


def calculate_predictions(data, model):
    preds = model.predict(x=data, verbose=1)
    return preds


if __name__ == "__main__":
    model_name = "model_v1_60s_focal_3.keras"
    model = keras.models.load_model(filepath=os.path.join("models", model_name), custom_objects={"SliceLayer": SliceLayer})
    data, labels = load_data(testing_data_path=os.path.join("data", "frames1", "testing.h5"))
    preds = calculate_predictions(data, model)

    data_dict = {
        "Test model": {"labels": labels, "preds": preds},
    }
    precision_recall_plot(data_dict)
    auc_roc_plot(data_dict)
    confusion_matrix_plot(data_dict)
