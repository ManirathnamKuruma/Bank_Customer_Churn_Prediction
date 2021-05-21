from flask import Flask, render_template, request
import jsonify
import requests
import joblib
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)
# load model and transformer
model = load_model('best_model.h5')
sc = joblib.load('scaler.pkl')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        cs = int(request.form['cs'])
        gender=request.form['gender']
        if (gender=='male'):
            gender=1
        else:
            gender=0
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['bal'])
        nprod=request.form['nprod']
        if (nprod==4):
            nprod=4
        elif (nprod==2):
            nprod=2
        elif (nprod==3):
            nprod=3
        else:
            nprod=1
        cc=request.form['cc']
        if (cc=='yes'):
            cc=1
        else:
            cc=0
        isactive = request.form['isactive']
        if (isactive == 'yes'):
            isactive = 1
        else:
            isactive = 0
        es = float(request.form['es'])
        geo_fr=0
        geo_ger=0
        geo_sp=0
        geo= request.form['geo']
        if (geo=='fr'):
            geo_fr=1
        elif (geo=='ger'):
            geo_ger=1
        else:
            geo_sp=1

        prediction=model.predict(sc.transform([[cs,gender,age,tenure,balance,nprod,cc,isactive,es,geo_fr,geo_ger,geo_sp]])) > 0.5
        output=prediction.tolist()

        if output==[[True]]:
            return render_template('index.html',prediction_text="Customer may stop doing the business")
        else:
            return render_template('index.html',prediction_text="Customer is likely to continue the business")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

