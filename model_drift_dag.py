from airflow import DAG
#from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import pickle
#from airflow.hooks.postgres_hook import PostgresHook
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift
#from airflow.hooks.postgres_hook import PostgresHook
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

    with open("newmodel.pkl", "wb") as f:
        pickle.dump(classifier, f)

def no_retrain_model():
    print("No model drift detected. No retraining needed.")

def load_reference_data():
    reference_data=pd.read_csv('prima_13.csv')
    return reference_data

def load_current_data():
    current_data=pd.read_csv('final.csv')
    return current_data
# Define your Airflow DAG
default_args = {
    'owner': 'user',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'conditional_model_drift_detection_dag',
    default_args=default_args,
    schedule_interval='@daily',  # Set your desired schedule interval
    catchup=False,
)

# Create a PythonOperator to execute the data drift detection function
detect_data_drift_task = PythonOperator(
    task_id='detect_model_drift_task',
    python_callable=detect_model_drift,
    provide_context=True,
    dag=dag,
)

# Use BranchPythonOperator to determine whether to retrain the model or not
branch_task = BranchPythonOperator(
    task_id='branch_task',
    python_callable=lambda ti: 'retrain_model_task' if ti.xcom_pull(task_ids='detect_model_drift_task') else 'no_retrain_model_task',
    provide_context=True,
    dag=dag,
)

# Define tasks for retraining the model and not retraining the model
retrain_model_task = PythonOperator(
    task_id='retrain_model_task',
    python_callable=retrain_model,
    dag=dag,
)

no_retrain_model_task = PythonOperator(
    task_id='no_retrain_model_task',
    python_callable=no_retrain_model,
    dag=dag,
)

# Set task dependencies
detect_data_drift_task >> branch_task
branch_task >> retrain_model_task
branch_task >> no_retrain_model_task

if __name__ == "__main__":
    dag.cli()
