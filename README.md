# Flight Fare Prediction:

![](https://github.com/RickyMehra29/Flight_Fare_Prediction_End_to_End_AWS/blob/main/Docs/FFP_GIF.gif)

The repository consists of files required for end to end implementation and deployment of Machine Learning Flight Fare Prediction web application created with Flask and deployed on the AWS platform.

![ML_Dev_Steps](https://github.com/RickyMehra29/Flight_Fare_Prediction_End_to_End_AWS/blob/main/Docs/ML_Dev_Steps.jpg)

![Model_Architecture](https://github.com/RickyMehra29/Flight_Fare_Prediction_End_to_End_AWS/blob/main/Docs/Model_Architecture.jpg)

The Flight Fare Prediction is a Flask web application to predict airline flight fares. The dataset for the project is taken from Kaggle, and it is a time-stamped dataset so, while building the model, extensive pre-processing was done on the dataset especially on the date-time columns to finally come up with a ML model which could effectively predict airline fares across various Indian Cities. The dataset had many features which had to pre-processed and transformed into new parameters for a cleaner and simple web application layout to predict the fares. The various independent features in the dataset were:

# Featured detail:
* Airline: The name of the airline
* Date_of_Journey: The date of the journey
* Source: The source from which the service begins.
* Destination: The destination where the service ends.
* Route: The route taken by the flight to reach the destination.
* Dep_Time: The time when the journey starts from the source.
* Arrival_Time: Time of arrival at the destination.
* Duration: Total duration of the flight.
* Total_Stops: Total stops between the source and destination.
* Additional_Info: Additional information about the flight
* Price: The price of the ticket

We have used the RandomForest Regressor to find out the predicted price of the flight for the given set of input. We are using the sample dataset from the Kaggle to develop the model and created both instance and batch prediction.

In instance prediction, user can provide input on the HTML webpage and on click of the predict button, the user will get to know the predicted fare of the flight for the given set of inputs.

In batch prediction, user can save .xlsx file in the Input_folder and can find the predicted prices in the Output_folder.


### 


### Step 1 - Install the requirements

```bash
pip install -r requirements.txt
```

### Step 2 - To Run .py files for prediction

```bash
python main.py for batch file prediction

python app.py for instance prediction



```
