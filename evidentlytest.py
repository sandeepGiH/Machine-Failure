import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evidently import ColumnMapping
from evidently import report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift
import pickle
import joblib
model = pickle.load(open('Naive_bayes.pkl', 'rb'))
imp_scale = joblib.load('imp_scale')
winsor = joblib.load('winsor')

def detect_model_drift():
    # Load reference and current datasets
    reference = load_reference_data()
    current = load_current_data()

    suite = TestSuite(tests=[
        TestColumnDrift('Downtime')
    ])

    suite.run(reference_data=reference, current_data=current)

    # Store the 'detected' value in a variable
    detected_value = suite.as_dict()['tests'][0]['parameters']['detected']

    return detected_value


def retrain_model():
    print("Model drift detected. Retraining the model...")

    # Load your dataset for model retraining
    data = load_current_data()
    data2 = data.drop(['Date', 'Machine_ID', 'Downtime'], axis = 1) # Excluding id column
    cleandata = pd.DataFrame(imp_scale.transform(data2), columns = imp_scale.get_feature_names_out())
    columns_list = list(cleandata.columns)
    cleandata[columns_list] = winsor.transform(cleandata[columns_list])
    
    y = data[['Downtime']]
    # Train a new RandomForestClassifier on the full dataset
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(cleandata, y)

    print("Model retrained.")

    with open("C:/Users/91818/Desktop/Machine Failure/newmodel.pkl", "wb") as f:
        pickle.dump(classifier, f)

def no_retrain_model():
    print("No model drift detected. No retraining needed.")

def load_reference_data():
    reference_data=pd.read_csv('prima_13.csv')
    return reference_data

def load_current_data():
    current_data=pd.read_csv('final.csv')
    return current_data

if detect_model_drift()==1:
	retrain_model()
else: 
    no_retrain_model()


