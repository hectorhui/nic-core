from datetime import datetime
from app.db import FireDB


class NIC_ML(object):
    def __init__(self, model_name, onlyeone=False, mode='smart'):
        """
        `mode`: 'smart' or 'api'
        """
        if(mode=='smart'):
            from app.dnn import Preprocess, detector, load_k_model
            paths = {
            'model_path': './app/model/models/' + model_name + '.json',
            'weights_path': './app/model/weights/' + model_name + '.h5',
            'meta_path':'./app/model/meta/' + model_name +  '.json',
            'labels_path':'./app/model/labels/labels.txt'
            }
            self.detector = detector(paths,onlyeone)
            if(onlyeone):
                self.preprocess = Preprocess(28595,140)
            else:
                self.preprocess = Preprocess(200,200)
        
        self.db = FireDB()
    
    def predict(self,data, threshold=0.2):
        data = self.preprocess.return_data(data)
        predictions = self.detector.predict(data)
        labels = self.detector.get_labels(predict=predictions,threshold=threshold)
        return labels
    
    #only for crawler
    def push_to_firedb(self, collection_name, data):
        try:
            today  = datetime.now().date().isoformat()
            self.db.insert(str(today) + '/' + collection_name + '/', data=data)
            return "Successfully save to FireDB"
        except :
            return "Something Wrong"
    #only for crawler
    def get_from_firedb(self,collection_name, by_date='', alldata=False):
        if(alldata == False):
            if by_date != '':
                date = by_date
            else:
                date = datetime.now().date().isoformat()
            try:
                data = self.db.get('/' + collection_name + '/' + str(date) + '/')
                return data
            except :
                return "Something wrong"
        elif(alldata):
            data = self.db.get(collection_name)
            return data
        
    def get_data_firedb(self, collection_name):
        return self.db.get(collection_name)
    
    def save_to_firedb(self, collection_name , newssite='', data=''):
        today  = datetime.now().date().isoformat()
        self.db.insert(collection_name +'/' + str(today) + '/' + newssite, data=data)
    
    def clean_db(self, collection_name):
        self.db.remove(collection_name)

