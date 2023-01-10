import pymongo
import pandas as pd
import json

from ffp.config import mongo_client
#from dotenv import load_dotenv
#load_dotenv()

DATA_FILE_NAME = "/config/workspace/Flight_Fare_Dataset.xlsx"
DATABASE_NAME = "flight_fare_db"
COLLECTION_NAME ="flight_fare_collection"

if __name__=="__main__":
    df = pd.read_excel(DATA_FILE_NAME)
    df.reset_index(drop=True,inplace=True)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json to dump these record in mongodb
    json_record = list(json.loads(df.T.to_json()).values())

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    print(f"{df.shape[0]} records have been uploaded in the Mongodb database-{DATABASE_NAME}")