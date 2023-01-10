from ffp.exception import FlightFareException
from ffp.logger import logging
from ffp.config import mongo_client
import pandas as pd
import numpy as np
import os, sys
import yaml
import dill



def get_collection_as_dataframe(database_name:str, collection_name:str)-> pd.DataFrame:
    try:
        logging.info(f"Reading data from Mongodb from database-{database_name} and collection-{collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        
        if "_id" in df.columns:
            logging.info("Removing _id column from the dataframe")
            df.drop("_id", axis=1, inplace =True)
        logging.info(f"Rows and colums: {df.shape}")
        return df

    except Exception as e:
        raise FlightFareException(e, sys)
    
def write_yaml_file(file_path, data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_obj:
            yaml.dump(data, file_obj)

    except Exception as e:
        raise FlightFareException(e, sys)

def save_object(file_path:str, obj:object)->None:
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"{file_path.split('/')[-1]} object has been saved thorough utils")
    except Exception as e:
        raise FlightFareException(e, sys)

def load_object(file_path:str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist.")
    
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise FlightFareException(e, sys)

def save_data(file_path:str, df)-> None:
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        df.to_csv(file_path, index= False, header =True)

    except Exception as e:
        raise FlightFareException(e, sys)

def load_data(file_path:str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The dataframe {file_path} does not exist.")

        return pd.read_csv(file_path)

    except Exception as e:
        raise FlightFareException(e, sys)




# Functions for data transformation

def split_date_feature(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    try:
        df['Date'] = df[column_name].str.split('/').str[0]
        df['Month']= df[column_name].str.split('/').str[1]
        df['Year'] = df[column_name].str.split('/').str[2]
        
        df['Date'] = df['Date'].astype(int)
        df['Month']= df['Month'].astype(int)
        df['Year'] = df['Year'].astype(int)
        
        df.drop(column_name, axis =1, inplace=True)
        logging.info(f"'{column_name}' column has been splitted into 'Date' and 'Month' columns") 
        return df

    except Exception as e:
        raise FlightFareException(e, sys)     

def split_time_feature(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    try:
        df[column_name] =df[column_name].str.split(' ').str[0]
    
        df[column_name+'_hour'] = df[column_name].str.split(':').str[0]
        df[column_name+'_min']  = df[column_name].str.split(':').str[1]

        df[column_name+'_hour' ] = df[column_name+'_hour'].astype(int)
        df[column_name+'_min']  = df[column_name+'_min'].astype(int) 
        
        df.drop(column_name, axis =1, inplace=True) 
        logging.info(f"'{column_name}' column has been splitted into '{column_name+'_hour'}' and '{column_name+'_min'}' columns")   
        return df

    except Exception as e:
        raise FlightFareException(e, sys)  

def split_duration_feature(df:pd.DataFrame, column_name:str)->pd.DataFrame:

    try:
        duration = list(df[column_name])

        for i in range(len(duration)):
            if len(duration[i].split(' '))==2:
                pass
            else:
                if 'h' in duration[i]:                  # Check if duration contains only hours
                    duration[i] = duration[i] + ' 0m'  # Adds 0 minutes
                else:
                    duration[i] ='0h '+ duration[i]    # Adds 0 hours, if only minutes available
        df[column_name] = duration               
        
        df[column_name+'_hour'] = df[column_name].str.split(' ').str[0].str.replace('h','')
        df[column_name+'_min']  = df[column_name].str.split(' ').str[1].str.replace('m','')

        df[column_name+'_hour'] = df[column_name+'_hour'].astype(int)
        df[column_name+'_min']  = df[column_name+'_min'].astype(int)  

        """if df[df[column_name+'_hour']==0].index != 0: 
            df.drop(df[df[column_name+'_hour']==0].index, axis=0, inplace =True) 
        """
        df.drop(column_name, axis =1, inplace=True)
        df.reset_index(drop=True) 
        logging.info(f"'{column_name}' column has been splitted into '{column_name+'_hour'}' and '{column_name+'_min'}' columns") 
        
        return df

    except Exception as e:
        raise FlightFareException(e, sys) 
