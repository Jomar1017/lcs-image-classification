import unittest
from skExSTraCSLocal import ExSTraCS,StringEnumerator
from EvaluationReporter import EvaluationReporter
from ModelEvaluator import ModelEvaluator
from pathlib import Path

FEATURES_DIR = Path("dataset/waste-image-dataset-extracted")
REPORTS_DIR = Path("reports")
train_file = FEATURES_DIR / "train_lbp_hog_pca_15.csv"
test_file = FEATURES_DIR / "test_lbp_hog_pca_15.csv"
class_map = FEATURES_DIR / "class_map.csv"

LOG_DIR = "MetaData/ExpRes/"
log_popfile_name = "log_population_test_lbp.csv"
log_trainingfile_name = "log_training_test_lbp.txt"

train_converter = StringEnumerator(train_file,'Class')
train_headers, train_classLabel, train_dataFeatures, train_dataPhenotypes = train_converter.get_params()
test_converter = StringEnumerator(test_file,'Class')
test_headers, test_classLabel, test_dataFeatures, test_dataPhenotypes = test_converter.get_params()

expert_knowledge = [1.0, 0.9063305617829138, 0.9207294223715887, 0.9763852509872054, 0.8080657558877218, 0.9360197289295088, 
                    0.7788220237990467, 0.8878705291211352, 0.644260832151975, 0.7600942394923319, 0.6274400325208331, 
                    0.6045143197604805, 0.722857464863158, 0.6449554490564906, 0.7020066542309898]
model = ExSTraCS(
    learning_iterations=1000,
    N=1000,
    nu=5,
    #chi=0.9,
    #mu=0.06,
    #theta_GA=50,
    #do_attribute_tracking=True,
    #do_attribute_feedback=True,
    expert_knowledge=expert_knowledge,
    log_dir=LOG_DIR,
    log_popfile_name=log_popfile_name,
    log_trainingfile_name=log_trainingfile_name
)

model.fit(train_dataFeatures, train_dataPhenotypes)
accuracy = model.score(test_dataFeatures,test_dataPhenotypes)
print(f"Accuracy: {accuracy}")

#Create reports of LCS
print("================= Creating LCS Reports =================")
prediction = model.predict(test_dataFeatures)
reporter = EvaluationReporter(y_true=test_dataPhenotypes,y_pred=prediction,output_dir=REPORTS_DIR,class_map=class_map)
reporter.save_classification_report("lcs_classification_report.txt", "LCS Classification Report")
reporter.save_confusion_matrix_csv("lcs_confusion_matrix.csv")
reporter.save_confusion_matrix_image("lcs_confusion_matrix.png")
reporter.save_learning_curve("MetaData/ExpRes/log_training_test_lbp.txt")

#Run other models (RF and SVM) for comparison
print("=============== Running RF and SVM Model ===============")
model_evaluator = ModelEvaluator(train_dataFeatures,test_dataFeatures,train_dataPhenotypes,test_dataPhenotypes)
model_evaluator.run_all_models()