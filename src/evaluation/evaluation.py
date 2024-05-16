import os
import re
import sys
import itertools
import h5py
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, f1_score
sys.path.append("src")
from model_training.architectures import SliceLayer


class ModelEvaluator:
    def __init__(self, model_names):
        """
        Initialize ModelEvaluator.

        Args:
        """
        self.data_dict = self._create_data_dict(model_names)

    def _create_data_dict(self, model_names):
        """ """
        os.makedirs("plots", exist_ok=True)
        data_dict = {}
        for model_name in model_names:
            loso_subject = model_name.split("_")[1]
            window_size = model_name.split("_")[2]
            model = keras.models.load_model(filepath=os.path.join("models", model_name), custom_objects={"SliceLayer": SliceLayer})
            data, data_labels = self._load_data(os.path.join("data", f"frames_{loso_subject}_{window_size}", "testing.h5"))
            preds = self._calculate_predictions(data, model)
            data_dict[model_name] = {"labels": data_labels, "preds": preds}
        return data_dict

    def _load_data(self, testing_data_path):
        """
        Load testing data from HDF5 file.

        Args:
            testing_data_path (str): Path to the HDF5 file containing testing data.

        Returns:
            np.array: Testing data.
            np.array: Labels.
        """
        with h5py.File(testing_data_path, "r") as file:
            dataset_names = list(file.keys())
            data = []
            labels = []
            for dataset in dataset_names:
                data.append(file[dataset][:])
                labels.append(file[dataset].attrs["label"])
        return np.array(data), np.array(labels)[:, np.newaxis]

    def _calculate_predictions(self, data, model):
        """
        Calculate predictions using the provided model.

        Args:
            data (np.array): Input data.
            model (keras.Model): Trained model.

        Returns:
            np.array: Predictions.
        """
        preds = model.predict(x=data, verbose=1)
        return preds

    def f1_score_table(self, data_dict):
        """
        Calculate F1 scores for each model in the data dictionary and save them in a table.

        Args:
            data_dict (dict): A dictionary containing model names as keys and data (labels and predictions) as values.
        """
        plt.figure(figsize=(8, 6))
        table_data = []
        for model_name, data in data_dict.items():
            labels = data["labels"]
            preds = data["preds"]
            f1 = f1_score(labels, np.round(preds))
            table_data.append([model_name, f1])

        plt.axis("tight")
        plt.axis("off")
        plt.table(cellText=table_data, colLabels=["Model", "F1 Score"], loc="center", cellLoc="center", colWidths=[0.3, 0.3])
        plt.savefig(os.path.join(self.plot_dir, "f1_scores.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def precision_recall_plot(self, data_dict):
        """
        Generate precision-recall curves for each model in the data dictionary.

        Args:
            data_dict (dict): A dictionary containing model names as keys and data (labels and predictions) as values.
        """
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
        plt.savefig(os.path.join(self.plot_dir, "precision_recall_curves.png"), dpi=300)

    def auc_roc_plot(self, data_dict):
        """
        Generate ROC curves for each model in the data dictionary.

        Args:
            data_dict (dict): A dictionary containing model names as keys and data (labels and predictions) as values.
        """
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
        plt.savefig(os.path.join(self.plot_dir, "roc_curves.png"), dpi=300)

    def confusion_matrix_plot(self, data_dict):
        """
        Generate confusion matrices for each model in the data dictionary.

        Args:
            data_dict (dict): A dictionary containing model names as keys and data (labels and predictions) as values.
        """
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
            plt.savefig(os.path.join(self.plot_dir, f"confusion_matrix_{model_name}.png"), dpi=300)


if __name__ == "__main__":
    evaluator = ModelEvaluator(model_names=[model_name for model_name in os.listdir("models") if re.search(r"v2", model_name)])
    evaluator.f1_score_table()
    evaluator.precision_recall_plot()
    evaluator.auc_roc_plot()
    evaluator.confusion_matrix_plot()
