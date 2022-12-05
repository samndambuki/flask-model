from flask import Flask, request
import pickle
from scipy.stats import mode
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open("./model/diseaseprediction.pickle", "rb"))
final_rf_model = pickle.load(open("./model/rf_model.pkl", "rb"))
final_nb_model = pickle.load(open("./model/nb_model.pkl", "rb"))
final_svm_model = pickle.load(open("./model/svm_model.pkl", "rb"))

data = pd.read_csv("./data/Training.csv").dropna(axis =1 )
labels = data["prognosis"]
X = data.iloc[:,:-1]
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

symptoms = X.columns.values 
symptom_index = {}
for index, value in enumerate (symptoms):
  symptom = " ".join([i.capitalize() for i in value.split("_")]) 
  symptom_index[symptom] = index

data_dict = {
    "symptom_index" : symptom_index,
    "predictions_classes" : encoder.classes_
}

@app.get("/")
def index():
    print(X.shape)
    return "Okay!!"


@app.post("/predict")
def predict():
    data = request.get_json()
    symptoms = data["symptoms"]
    pred = predictDisease(symptoms)
    print(pred)
    return pred["final_prediction"]


def predictDisease(symptoms):
  symptoms = symptoms.split(",")

  #Creating input data for the models
  input_data = [0] * len(data_dict["symptom_index"])
  for symptom in symptoms:
    index = data_dict["symptom_index"][symptom]
    input_data[index] = 1

  #reshaping the iinput data and converting it 
  #into suitable format for model predictions
  input_data  = np.array(input_data).reshape(1,-1)

  #generating individual ouptuts
  rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
  nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
  svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

  #Making final prediction b taking mode of all pedictions
  final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
  predictions = {
      "rf_model_prediction":rf_prediction,
      "naive_bayes_prediction":nb_prediction,
      "svm_model_prediction":svm_prediction,
      "final_prediction":final_prediction
  }
  return predictions



if __name__ == "__main__":
    app.run(debug=True)