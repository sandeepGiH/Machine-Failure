# Import libraries
from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib

reference=pd.read_csv(r"C:\Users\91818\Desktop\Machine Failure\prima_13.csv")
model = pickle.load(open('newmodel.pkl', 'rb'))
imp_scale = joblib.load('imp_scale')
winsor = joblib.load('winsor')

import evidently
from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        
        try:
            data = pd.read_csv(f)
        except:
                try:
                    data = pd.read_excel(f)
                except:      
                    data = pd.DataFrame(f)       
        # Drop the unwanted features
        data2 = data.drop(['Date', 'Machine_ID'], axis = 1) # Excluding id column
        cleandata = pd.DataFrame(imp_scale.transform(data2), columns = imp_scale.get_feature_names_out())
        columns_list = list(cleandata.columns)
        cleandata[columns_list] = winsor.transform(cleandata[columns_list])
        predictions = pd.DataFrame(model.predict(cleandata.values), columns = ['DownTime'])
        final = pd.concat([predictions, data], axis = 1)
        final.to_csv('final.csv',index=False)
        html_table = final.to_html(classes = 'table table-striped')
        
        return render_template("result.html", y = html_table)

@app.route('/monitor', methods = ['POST'])
def monitoring():
    if request.method == 'POST':
        
            report = Report(metrics=[
                DataDriftPreset(),
            ])
            current=pd.read_csv("final1.csv")
            report.run(reference_data=reference, current_data=current)
            report.save_html("datadrift.html")  # Save the HTML report as "datadrift.html"
            
            # Return the saved HTML file as a response
            return "html file downloaded"
        
if __name__=='__main__':
    #from mod import FeatureDropper
    app.run(debug = True)
