from ffp.exception import FlightFareException
from ffp.logger import logging
from ffp.predictor import ModelResolver
from ffp import utils
import os, sys
import pandas as pd 
import numpy as np 
from datetime import datetime

from ffp.utils import load_object
from ffp.components.data_ingestion import DataIngestion
from ffp.components.data_transformation import DataTransformation
from ffp.entity import artifact_entity, config_entity

PREDICTION_DIR = "Prediction_Output"

reset_cols = ['Date', 'Month', 'Dep_Time_hour', 'Dep_Time_min', 'Arrival_Time_hour','Arrival_Time_min', 
                    'Duration_hour','Duration_min','Airline','Source', 'Destination','Total_Stops', 'Additional_Info']

Airline_TRANSFORMER_OBJECT_FILE_NAME= "Airline_transformer.pkl"
Source_Destination_TRANSFORMER_OBJECT_FILE_NAME= "Source_Destination_transformer.pkl"
Total_Stops_TRANSFORMER_OBJECT_FILE_NAME= "Total_Stops_transformer.pkl"
Additional_Info_TRANSFORMER_OBJECT_FILE_NAME= "Additional_Info_transformer.pkl"

def start_batch_prediction(input_file_path):
    try:

        logging.info(f"Reading input file :{input_file_path}")
        input_df = pd.read_excel(input_file_path)
        print(input_df.shape)


        prediction_df = input_df.copy(deep=True)

        prediction_df['Destination'] = prediction_df['Destination'].replace("New Delhi", "Delhi")
        prediction_df['Additional_Info'] = prediction_df['Additional_Info'].replace('No Info', 'No info')
        
        prediction_df = utils.split_date_feature(df=prediction_df, column_name='Date_of_Journey')
        prediction_df = utils.split_time_feature(df=prediction_df, column_name='Dep_Time')
        prediction_df = utils.split_time_feature(df=prediction_df, column_name='Arrival_Time')
        prediction_df = utils.split_duration_feature(df=prediction_df, column_name='Duration')

        #loading model & transformer objects
        logging.info(f"Creating model resolver object to load Latest model & transformer objects")
        model_resolver = ModelResolver(model_registry="saved_models")

        Airline_transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path(Airline_TRANSFORMER_OBJECT_FILE_NAME))
        Source_Destination_transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path(Source_Destination_TRANSFORMER_OBJECT_FILE_NAME))
        Total_Stops_transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path(Total_Stops_TRANSFORMER_OBJECT_FILE_NAME))
        Additional_Info_transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path(Additional_Info_TRANSFORMER_OBJECT_FILE_NAME))

        print(Airline_transformer)

        prediction_df['Airline'] = prediction_df['Airline'].map(Airline_transformer)
        prediction_df['Source'] = prediction_df['Source'].map(Source_Destination_transformer)
        prediction_df['Destination'] = prediction_df['Destination'].map(Source_Destination_transformer)
        prediction_df['Total_Stops'] = prediction_df['Total_Stops'].map(Total_Stops_transformer)
        prediction_df['Additional_Info'] = prediction_df['Additional_Info'].map(Additional_Info_transformer)

        prediction_df.drop(['Route', 'Year'], axis =1, inplace = True)      
        prediction_df = prediction_df.reindex(reset_cols, axis =1) 

        logging.info(f"Replacing 'na' values with np.NAN")
        prediction_df.replace(to_replace="na",value=np.NAN,inplace=True)

        print(prediction_df.columns, "\n", prediction_df.shape)
        prediction_df.to_excel("/config/workspace/Flight_check_prediction_df.xlsx", index= False, header =True)

        logging.info(f"Loading model to make prediction")
        model = utils.load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(prediction_df)
       
        input_df["Prediction"] = prediction
        input_df["Prediction"] = round(input_df["Prediction"],0)

        os.makedirs(PREDICTION_DIR, exist_ok=True)

        prediction_file_name = os.path.basename(input_file_path).replace(".xlsx",f"{datetime.now().strftime('%m-%-d%Y__%H:%M:%S')}.xlsx")
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name )

        input_df.to_excel(prediction_file_path, index=False, header=True)
        return prediction_file_path

    except Exception as e:
        raise FlightFareException(e, sys)


