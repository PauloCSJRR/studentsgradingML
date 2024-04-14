from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path # type: ignore
sys.path.append(str(Path(__file__).parent.parent.parent))   

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app=application

# Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),  # Corrected field name
            writing_score=int(request.form.get('writing_score'))  # Corrected field name
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # For debugging
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        if results:  # Check if results are not empty
            return render_template('home.html', results=results[0])
        else:
            return render_template('home.html', error_message="Prediction failed.")

    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)