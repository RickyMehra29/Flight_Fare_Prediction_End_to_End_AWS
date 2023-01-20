from ffp.pipeline.training_pipeline import start_training_pipeline
from ffp.pipeline.batch_prediction import start_batch_prediction
from ffp.exception import FlightFareException
import os, sys

file_path="/config/workspace/input_files/Test_dataset.xlsx"
#print(__name__)

if __name__=="__main__":
    try:
        start_point = start_training_pipeline()
        print(start_point)

        #output_file = start_batch_prediction(input_file_path=file_path)
        #print(output_file)

    except Exception as e:
       raise FlightFareException(e, sys)