from sklearn.svm import SVC #SVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class ModelEvaluator:
    def __init__(self, train_dataFeatures, test_dataFeatures, train_dataPhenotypes, test_dataPhenotypes):
        self.train_dataFeatures = train_dataFeatures
        self.test_dataFeatures = test_dataFeatures
        self.train_dataPhenotypes = train_dataPhenotypes
        self.test_dataPhenotypes = test_dataPhenotypes
        self.scaler = StandardScaler()
        self.output_dir = Path("reports")

    def scale_features(self):
        self.scaled_train_features = self.scaler.fit_transform(self.train_dataFeatures)
        self.scaled_test_features = self.scaler.transform(self.test_dataFeatures)

    def run_random_forest(self):
        self.scale_features() #Scale the features
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(self.scaled_train_features, self.train_dataPhenotypes) #Train RF model

        rf_pred = rf_model.predict(self.scaled_test_features) #Predict accuracy score
        rf_accuracy = accuracy_score(self.test_dataPhenotypes, rf_pred)
        # rf_precision = precision_score(self.test_dataPhenotypes, rf_pred, average='weighted')
        # rf_recall = recall_score(self.test_dataPhenotypes, rf_pred, average='weighted')
        # rf_f1 = f1_score(self.test_dataPhenotypes, rf_pred, average='weighted')

        filename = "rf_classification_report.txt"
        report = classification_report(self.test_dataPhenotypes, rf_pred, zero_division=0)
        report_path = self.output_dir / filename
        with report_path.open("w") as f:
            f.write(f"Random Forest Classification Report\n")
            f.write(report)
        print(f"RF Classification file: {filename} saved in {self.output_dir} folder with accuracy: {rf_accuracy}.")
    
    def run_SVM(self):
        self.scale_features() #Scale the features
        svm_model = SVC() #Initialize SVM model
        svm_model.fit(self.scaled_train_features, self.train_dataPhenotypes) #Train SVM model

        svm_prediction = svm_model.predict(self.scaled_test_features)
        svm_accuracy = accuracy_score(self.test_dataPhenotypes, svm_prediction)
        # precision = precision_score(self.test_dataPhenotypes, svm_prediction, average='weighted')
        # recall = recall_score(self.test_dataPhenotypes, svm_prediction, average='weighted')
        # f1 = f1_score(self.test_dataPhenotypes, svm_prediction, average='weighted')

        filename = "svm_classification_report.txt"
        report = classification_report(self.test_dataPhenotypes, svm_prediction, zero_division=0)
        report_path = self.output_dir / filename
        with report_path.open("w") as f:
            f.write(f"SVM Classification Report\n")
            f.write(report)
        print(f"SVM Classification file: {filename} saved in {self.output_dir} folder with accuracy: {svm_accuracy}")

    def run_all_models(self):
        self.run_random_forest()
        self.run_SVM()