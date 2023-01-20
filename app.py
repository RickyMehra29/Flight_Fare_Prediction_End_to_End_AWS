from ffp.pipeline.training_pipeline import start_training_pipeline
from ffp.pipeline.batch_prediction import start_batch_prediction
from ffp.exception import FlightFareException
from ffp.logger import logging
from ffp.entity import config_entity
from ffp import utils
import os, sys

from ffp.predictor import ModelResolver

import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import datetime as dt

app = Flask(__name__)

logging.info(f"Loading Latest model from saved_models folder!!")
latest_dir = ModelResolver().get_latest_dir_path()

model_path= os.path.join(latest_dir,"model",config_entity.MODEL_FILE_NAME)
flight_model = utils.load_object(file_path=model_path)

logging.info(f"Loading Latest trasnformers from saved_models folder!!")

Airline_path = os.path.join(latest_dir,"transformer",config_entity.Airline_TRANSFORMER_OBJECT_FILE_NAME)
Source_Destination_path = os.path.join(latest_dir,"transformer",config_entity.Source_Destination_TRANSFORMER_OBJECT_FILE_NAME)
Total_Stops_path = os.path.join(latest_dir,"transformer",config_entity.Total_Stops_TRANSFORMER_OBJECT_FILE_NAME)
Additional_Info_path = os.path.join(latest_dir,"transformer",config_entity.Additional_Info_TRANSFORMER_OBJECT_FILE_NAME)

Airline_dict = utils.load_object(Airline_path)
Source_Destination_dict = utils.load_object(Source_Destination_path)
Total_Stops_dict = utils.load_object(Total_Stops_path)
Additional_Info_dict = utils.load_object(Additional_Info_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = [x for x in request.form.values()]
    depart_time = data[0].split('T')[1]
    result = dt.datetime.strptime(data[1], '%H:%M')-dt.datetime.strptime(depart_time,'%H:%M')
    print(result.seconds/60)

    if dt.datetime.strptime(depart_time,'%H:%M')==dt.datetime.strptime(data[1], '%H:%M'):
        duration_h =24
        duration_min = 0
    else:
        duration_h = (result.seconds/60)//60
        duration_min = (result.seconds/60)%60
    print(duration_h, duration_min)

    filtered_data = []

    filtered_data.append(data[0].split('T')[0].split('-')[2])
    filtered_data.append(data[0].split('T')[0].split('-')[1])
    filtered_data.append(data[0].split('T')[1].split(':')[0])
    filtered_data.append(data[0].split('T')[1].split(':')[1])
    filtered_data.append(data[1].split(':')[0])
    filtered_data.append(data[1].split(':')[1])
    filtered_data.append(duration_h)
    filtered_data.append(duration_min)
    filtered_data.append(data[2])
    filtered_data.append(data[3])
    filtered_data.append(data[4])
    filtered_data.append(data[5])
    filtered_data.append(data[6])
    print(filtered_data)

    filtered_data[8] = int((pd.Series(filtered_data[8]).map(Airline_dict)).values)
    filtered_data[9] = int((pd.Series(filtered_data[9]).map(Source_Destination_dict)).values)
    filtered_data[10] = int((pd.Series(filtered_data[10]).map(Source_Destination_dict)).values)
    filtered_data[11] = int((pd.Series(filtered_data[11]).map(Total_Stops_dict)).values)
    filtered_data[12] = int((pd.Series(filtered_data[12]).map(Additional_Info_dict)).values)

    print(filtered_data)

    filtered_data = [int(x) for x in filtered_data]
    final_input= np.array(filtered_data).reshape(1,-1)
    print(final_input)

    output = flight_model.predict(final_input)[0]
    print(output)
  
    return render_template('home.html', output_text="The Price of the fight is {}.".format(round(output,2)))
    
if __name__=="__main__":
    try:
        #app.run(debug=True)
        app.run(host="0.0.0.0")
        #app.run(host="0.0.0.0", port=8000)

    except Exception as e:
       raise FlightFareException(e, sys)