
import gradio as gr
import pickle
def make_prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    with open("knn_pickle.pkl", "rb") as f:
        model  = pickle.load(f)
        preds = model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    if preds == 1:
            return "patient having diabetese"
    return "patient does not have diabetese"

#Create the input component for Gradio since we are expecting 4 inputs

Pregnancies=gr.Number(label = "Preg_number")
Glucose= gr.Number(label = "glucose level")
BP = gr.Number(label = "BP Level")
SkinThickness = gr.Number(label = "Pkin thickness")
Insulin = gr.Number(label = "insulin level")
BMI= gr.Number(label = "BMI Level")
Dpf = gr.Number(label = "dpf")
Age= gr.Number(label = "Patients age")
# We create the output
output = gr.Textbox()


app = gr.Interface(fn = make_prediction, inputs=[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, Dpf, Age], outputs=output)
app.launch()

