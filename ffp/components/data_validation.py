from ffp.exception import FlightFareException
from ffp.logger import logging
from ffp.entity import config_entity, artifact_entity
from ffp import utils
import os, sys
import pandas as pd
import numpy as np
from typing import Optional

from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self,
                    data_validation_config: config_entity.DataValidationConfig,
                    data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*10} Stage- 02 Data Validation initiated {'<<'*10}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact

            self.validation_error_dict = dict()
            self.TARGET_COLUMN="Price"

        except Exception as e:
            raise FlightFareException(e, sys)

    def missing_values_in_columns(self, df, report_key_name)->Optional[pd.DataFrame]:
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]

            drop_column_names = null_report[null_report>threshold].index
            if len(list(drop_column_names))>0:
                df.drop(list(drop_column_names), axis=1, inplace=True)
                logging.info(f"Below features have missing values greater than threshold-{threshold} & have been removed \n{list(drop_column_names)}")

                self.validation_error_dict[report_key_name] = list(drop_column_names)

            #return None if no columns left
            if len(df.columns)==0:
                return None
                logging.info(f"All features have missing values greater than threshold-{threshold}, kindly review threshold")
            
            logging.info(f"Missing values as per threshold is validated")
            return df            

        except Exception as e:
            raise FlightFareException(e, sys)
    
    def is_required_columns_exist(sef, current_df, base_df, report_key_name)->bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"{base_column} is not available in the current data.")
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error_dict[report_key_name]=missing_columns
                return False
                logging.info(f"Dataset does not have all required columns, kindly review columns.")
            logging.info(f"All expected features are available in the dataset")
            return True

        except Exception as e:
            raise FlightFareException(e, sys)

    def validate_data_types(self, current_df, base_df, report_key_name)->bool:
        try:         
            base_dtypes = base_df.dtypes            
            current_dtypes = current_df.dtypes
            current_columns = current_df.columns

            invalid_datatype_columns = []
            for idx, data_type in enumerate(base_dtypes):
                if data_type != current_dtypes[idx]:
                    invalid_datatype_columns.append(current_columns[idx])

            if len(invalid_datatype_columns)>0:
                self.validation_error_dict[report_key_name]=invalid_datatype_columns
                return False
                logging.info(f"{invalid_datatype_columns} columns don't have data types as per in the base dataset.")

            logging.info(f"Features have data types as per the base dataset.")
            return True

        except Exception as e:
            raise FlightFareException(e, sys)
    
    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info("Reading base data file")
            base_df = pd.read_excel(self.data_validation_config.base_file_path)
            base_df = self.missing_values_in_columns(df=base_df, report_key_name="Missing_value_within_base_dataset")

            logging.info("-----Validating Train dataset-----")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)            
            train_df = self.missing_values_in_columns(df=train_df, report_key_name="Missing_value_within_train_dataset")
            train_df_columns_status = self.is_required_columns_exist(current_df=train_df, base_df=base_df, report_key_name="Missing_columns_within_train_dataset")

            if train_df_columns_status:
                logging.info("All features are available in the train dataset, proceeing with their data types.")
                self.validate_data_types(current_df=train_df, base_df=base_df, report_key_name="Data_types_within_train_dataset")


            logging.info("-----Validating Test dataset-----")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df = self.missing_values_in_columns(df=test_df, report_key_name="Missing_value_within_test_dataset")
            test_df_columns_status = self.is_required_columns_exist(current_df=test_df, base_df=base_df, report_key_name="Missing_columns_within_test_dataset")

            if test_df_columns_status:
                logging.info("All features are available in the test dataset, proceeing with their data types.")
                self.validate_data_types(current_df=test_df, base_df=base_df, report_key_name="Data_types_within_test_dataset")

            # Write validation report
            logging.info("Writing validation report in yaml file")
            utils.write_yaml_file(file_path= self.data_validation_config.report_file_path, data= self.validation_error_dict)

            # Validation Artifact
            data_validation_artifact = artifact_entity.DataValidationArtifact(self.data_validation_config.report_file_path)
            logging.info(f"Data validation artifact: {data_validation_artifact}\n")
           
            return data_validation_artifact

        except Exception as e:
            raise FlightFareException(e, sys)



