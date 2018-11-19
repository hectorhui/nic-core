from flask import Flask, jsonify, request , Response
from app import NIC_ML

app = Flask(__name__)
model_name = 'reuters'
nic_ml = NIC_ML(model_name, True)

@app.route('/')
def main():
    return "Welcome to NIC Machine learning model API with Flask....."


@app.route('/gettoday/', methods=['GET'])
def get_today():
    data = nic_ml.get_from_firedb('nic-concepts')
    return jsonify(data)


@app.route('/bydate/')
@app.route('/bydate/date=<string:date>')
def get_bydate(date=''):
    data = nic_ml.get_from_firedb('nic-concepts', by_date=date)
    return jsonify(data)

@app.route('/predict/data=<string:data>')
def predict(data=''):
    data =[data]
    result = nic_ml.predict(data,0.4)
    if(len(result[0]) > 0):
        return jsonify(predictions=result, message=data[0])
    else:
        result = {
            "message": "no lables found"
        }
        return jsonify(result=result,message=data[0])

if __name__ == "__main__":
    app.run(debug=True)
