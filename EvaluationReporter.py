from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class EvaluationReporter:
    def __init__(self, y_true, y_pred, output_dir, class_map):
        self.y_true = y_true #ground truth
        self.y_pred = y_pred #predicted
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        class_map_df = pd.read_csv(class_map)
        self.class_labels = class_map_df.sort_values('class_id')['class_name'].tolist()  
    
    def save_classification_report(self, filename, title):
        report = classification_report(self.y_true, self.y_pred, zero_division=0)
        report_path = self.output_dir / filename
        with report_path.open("w") as f:
            f.write(f"{title}\n")
            f.write(report)
        print(f"Classification file: {filename} saved in {self.output_dir} folder.")

    def save_confusion_matrix_csv(self, filename):
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_df = pd.DataFrame(cm, index=self.class_labels, columns=self.class_labels)
        csv_path = self.output_dir / filename
        cm_df.to_csv(csv_path)
        print(f"Confusion Matrix csv: {filename} saved in reports folder.")

    def save_confusion_matrix_image(self, filename):
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_labels,
                    yticklabels=self.class_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        image_path = self.output_dir / filename
        plt.savefig(image_path)
        plt.close()
        print(f"Confusion Matrix image: {filename} saved in reports folder.")
    
    def save_learning_curve(self, log_file):
        df = pd.read_csv(log_file, header=None, delim_whitespace=True)
        df.columns = ['IterationNo', 'Accuracy', 'PopulationNumerosity', 'Population']
        #Remove last row
        df = df[pd.to_numeric(df['IterationNo'], errors='coerce').notnull()]
        df['IterationNo'] = pd.to_numeric(df['IterationNo'])
        df['Accuracy'] = pd.to_numeric(df['Accuracy'])

        #Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df['IterationNo'], df['Accuracy'], marker='o', linestyle='-')
        plt.title("LCS Accuracy Over Training Iterations")
        plt.xlabel("Iteration Number")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/learning_curve.png", dpi=300)

    def save_all(self):
        self.save_classification_report(filename="classification_report", title="CR")
        self.save_confusion_matrix_image(filename="confusion_matrix.png")
        self.save_confusion_matrix_csv(filename="confusion_matrix.txt")
        self.save_incorrect_predictions()