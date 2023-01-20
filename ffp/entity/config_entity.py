from ffp.exception import FlightFareException
from ffp.logger import logging
import os, sys

from datetime import datetime



DATABASE_NAME = "flight_fare_db"
COLLECTION_NAME ="flight_fare_collection"

FILE_NAME = "dataset.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TARGET_COLUMN = "Price"

MODEL_FILE_NAME = "model.pkl"

Airline_TRANSFORMER_OBJECT_FILE_NAME= "Airline_transformer.pkl"
Source_Destination_TRANSFORMER_OBJECT_FILE_NAME= "Source_Destination_transformer.pkl"
Total_Stops_TRANSFORMER_OBJECT_FILE_NAME= "Total_Stops_transformer.pkl"
Additional_Info_TRANSFORMER_OBJECT_FILE_NAME= "Additional_Info_transformer.pkl"


class TrainingPipelineConfig:
    
    def __init__(self):
        try: 
            self.artifact_dir = os.path.join(os.getcwd(), "Artifact", f"{datetime.now().strftime('%Y-%m-%d__%H:%M:%S')}")

        except Exception as e:
            raise FlightFareException(e,sys) 

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")

            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, TEST_FILE_NAME)    

            self.test_size = 0.2

        except Exception  as e:
            raise FlightFareException(e,sys) 


class DataValidationConfig:
     def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
            self.report_file_path = os.path.join(self.data_validation_dir, "validation_report.yaml")

            self.missing_threshold:float = 0.2
            self.base_file_path = "/config/workspace/Flight_Fare_Dataset.xlsx"     

        except Exception as e:
            raise FlightFareException(e, sys)

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")

        self.Airline_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformer", Airline_TRANSFORMER_OBJECT_FILE_NAME)
        self.Source_Destination_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformer", Source_Destination_TRANSFORMER_OBJECT_FILE_NAME)
        self.Total_Stops_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformer", Total_Stops_TRANSFORMER_OBJECT_FILE_NAME)
        self.Additional_Info_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformer", Additional_Info_TRANSFORMER_OBJECT_FILE_NAME)

        self.transformed_train_path = os.path.join(self.data_transformation_dir, "Transformed", TRAIN_FILE_NAME)
        self.transformed_test_path = os.path.join(self.data_transformation_dir, "Transformed", TEST_FILE_NAME)

class ModelTrainerConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir , "model_trainer")

        self.model_path = os.path.join(self.model_trainer_dir,"model", MODEL_FILE_NAME)
        self.expected_error = 2000
        self.overfitting_threshold = 200

class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold = 200
        

class ModelPusherConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
        self.saved_model_dir = os.path.join("saved_models") 
        self.pusher_model_dir = os.path.join(self.model_pusher_dir, "saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir, MODEL_FILE_NAME)


        self.Airline_pusher_transformer_path = os.path.join(self.model_pusher_dir, "transformer", Airline_TRANSFORMER_OBJECT_FILE_NAME)
        self.Source_Destination_pusher_transformer_path = os.path.join(self.model_pusher_dir,"transformer", Source_Destination_TRANSFORMER_OBJECT_FILE_NAME)
        self.Total_Stops_pusher_transformer_path = os.path.join(self.model_pusher_dir,"transformer", Total_Stops_TRANSFORMER_OBJECT_FILE_NAME)
        self.Additional_Info_pusher_transformer_path = os.path.join(self.model_pusher_dir,"transformer", Additional_Info_TRANSFORMER_OBJECT_FILE_NAME)

