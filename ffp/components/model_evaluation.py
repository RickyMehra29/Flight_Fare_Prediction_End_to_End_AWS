from ffp.predictor import ModelResolver
from ffp.exception import FlightFareException
from ffp.logger import logging
from ffp.entity import config_entity, artifact_entity
from ffp import utils
import pandas as pd
import sys,os

from sklearn.metrics import mean_absolute_error
from ffp.entity.config_entity import TARGET_COLUMN

class ModelEvaluation:
    def __init__(self, 
                    model_eval_config: config_entity.ModelEvaluationConfig,
                    data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                    data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                    model_trainer_artifact: artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*10} Stage-05 Model Evaluation Initiated {'<<'*10}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise FlightFareException(e,sys)

    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info("if 'saved_model' folder has model then we will compare which model is best btw currently trained model or" 
                "latest model available in the 'saved_folder'")

            latest_dir_path = self.model_resolver.get_latest_dir_path()
            
            if latest_dir_path is None:
                logging.info(f"Previous model is not available.")
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_error=None)
                logging.info(f"Model Evaluation Artifact--- {model_eval_artifact}\n")
                return model_eval_artifact
            
            logging.info("Finding location of latest model_path")  
            model_path = self.model_resolver.get_latest_model_path()

            #Previous trained model objects
            logging.info("Previously trained model objects")
            previous_model= utils.load_object(file_path= model_path)

            print(type(previous_model))
            print(previous_model)

            test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_path)
            x_test, y_true = test_df.iloc[:,:-1], test_df.iloc[:,-1] 

            # Finding MAE using previously trained model
            y_pred = previous_model.predict(x_test)
            previous_model_error = mean_absolute_error(y_true, y_pred)
            logging.info(f"MAE using previously trained model: {previous_model_error}")


            #Currently trained model objects
            logging.info("Currently trained model objects")            
            current_model  = utils.load_object(file_path=self.model_trainer_artifact.model_path)

            y_pred = current_model.predict(x_test)
            current_model_error = mean_absolute_error(y_true, y_pred)
            logging.info(f"MAE using currently trained model: {current_model_error}")
            

            if current_model_error >= previous_model_error:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_error= previous_model_error - current_model_error)

            logging.info(f"Model eval artifact: {model_eval_artifact}\n")
            return model_eval_artifact

        except Exception as e:
            raise FlightFareException(e,sys)



          