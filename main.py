from flask import Flask, jsonify, request , Response
from app import NIC_ML

app = Flask(__name__)
model_name = 'reuters'
nic_ml = NIC_ML(model_name, True,mode='api')

@app.route('/')
def main():
    nic_core_api = {
        "Message":"Welcome to NIC Machine learning model API with Flask.....",
        "host/gettoday/": "To get all Process-news from today",
        "host/bydate/<date>" : "To get all processed-news from <date>",
        "host/predict/data=<data>" : "To make a prediction on <data> ",
        "host/concepts/" :"To get all available concepts with its related news links",
        "host/concepts/<date>" :"To get all available concepts with its related news links within <date>",
        "host/concept/<concept_name>" :"To get all available news links from <concept_name>",
        "host/concept/<concept_name>/<date>" :"To get all  news links with <concept_name> form date"

    }
    return jsonify(nic_core_api)


@app.route('/gettoday/', methods=['GET'])
def get_today():
    data = nic_ml.get_from_firedb('nic-concepts')
    return jsonify(data)

@app.route('/bydate/<string:date>')
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

@app.route('/concepts/')
@app.route('/concepts/<date>')
def all_concepts(date=''):
    data = nic_ml.get_data_firedb('concepts')
    if(date !=''):
        date = '' if date == 'today' else date
        all_data = {}
        for concept in data:
            all_data[concept]={}
            concept_data = nic_ml.get_from_firedb('concepts/'+  concept + '/', by_date=date) 
            all_data[concept] = concept_data if concept_data != None else {"message": "No data for " + concept + ' on ' + date}
        data = all_data
    return jsonify(data)

@app.route('/concept/<concept_name>/')
@app.route('/concept/<concept_name>/<date>')
def concept(concept_name='', date=''):
    alldata = False if date == 'today' else True
    if alldata == False:
        date=''
    data = nic_ml.get_from_firedb('concepts/' + concept_name + '/' , by_date=date, alldata=alldata)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
