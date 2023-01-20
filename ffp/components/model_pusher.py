from ffp.predictor import ModelResolver
from ffp.exception import FlightFareException
from ffp.logger import logging
from ffp.entity import artifact_entity, config_entity
from ffp.utils import load_object, save_object
import os,sys

class ModelPusher:
    def __init__(self, model_pusher_config: config_entity.ModelPusherConfig,
                        data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                        model_trainer_artifact : artifact_entity.ModelTrainerArtifact):
        
        try:
            logging.info(f"{'>>'*10} Stage 06- Model Pusher Initiated {'<<'*10}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)

        except Exception as e:
            raise FlightFareException(e, sys)

    def initiate_model_pusher(self)-> artifact_entity.ModelPusherArtifact:
        try:
            #loading objects
            logging.info(f"Loading transformer dictionaries and model")
            Airline_transformer = load_object(file_path=self.data_transformation_artifact.Airline_transformer_object_path)
            Source_Destination_transformer = load_object(file_path=self.data_transformation_artifact.Source_Destination_transformer_object_path)
            Total_Stops_transformer = load_object(file_path=self.data_transformation_artifact.Total_Stops_transformer_object_path)
            Additional_Info_transformer = load_object(file_path=self.data_transformation_artifact.Additional_Info_transformer_object_path)

            model = load_object(file_path=self.model_trainer_artifact.model_path)

            # model pusher dir for local network
            logging.info(f"Saving models into model pusher directory")
            save_object(file_path=self.model_pusher_config.Airline_pusher_transformer_path, obj=Airline_transformer)
            save_object(file_path=self.model_pusher_config.Source_Destination_pusher_transformer_path, obj=Source_Destination_transformer)
            save_object(file_path=self.model_pusher_config.Total_Stops_pusher_transformer_path, obj=Total_Stops_transformer)
            save_object(file_path=self.model_pusher_config.Additional_Info_pusher_transformer_path, obj=Additional_Info_transformer)

            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)


            # saved_models dir for S3 Bucket
            logging.info(f"Saving model in saved_models dir for S3 Bucket")
            
            model_path=self.model_resolver.get_latest_save_model_path()
            save_object(file_path=model_path, obj=model)



            logging.info(f"Model path is {model_path}")


            Airline_path = self.model_resolver.get_latest_save_transformer_path(config_entity.Airline_TRANSFORMER_OBJECT_FILE_NAME)
            Source_Destination_path = self.model_resolver.get_latest_save_transformer_path(config_entity.Source_Destination_TRANSFORMER_OBJECT_FILE_NAME)
            Total_Stops_path = self.model_resolver.get_latest_save_transformer_path(config_entity.Total_Stops_TRANSFORMER_OBJECT_FILE_NAME)
            Additional_Info_path = self.model_resolver.get_latest_save_transformer_path(config_entity.Additional_Info_TRANSFORMER_OBJECT_FILE_NAME)

            save_object(file_path=Airline_path, obj=Airline_transformer)
            save_object(file_path=Source_Destination_path, obj=Source_Destination_transformer)
            save_object(file_path=Total_Stops_path, obj=Total_Stops_transformer)
            save_object(file_path=Additional_Info_path, obj=Additional_Info_transformer)        


            model_pusher_artifact = artifact_entity.ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.model_pusher_dir,
                                                        saved_model_dir=self.model_pusher_config.model_pusher_dir)

            logging.info(f"Model pusher artifact: {model_pusher_artifact}\n")
            return model_pusher_artifact

        except Exception as e:
            raise FlightFareException(e, sys)