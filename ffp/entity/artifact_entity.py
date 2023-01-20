from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    data_validation_artifact:str


@dataclass
class DataTransformationArtifact:
    transformed_train_path:str
    transformed_test_path:str
    Airline_transformer_object_path:str
    Source_Destination_transformer_object_path:str
    Total_Stops_transformer_object_path:str
    Additional_Info_transformer_object_path:str

@dataclass
class ModelTrainerArtifact:
    model_path:str
    mae_train_error:float 
    mae_test_error:float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_error:float

@dataclass
class ModelPusherArtifact:
    pusher_model_dir:str    
    saved_model_dir:str     
