import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from app.util import read_file, load_data
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from nltk.metrics import ConfusionMatrix
from keras.utils import to_categorical , plot_model
from keras import backend as K
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Flatten, Embedding
from keras.optimizers import Adam
import tensorflow as tf


def get_model(max_feature, num_features):
    print('Max_feature:',max_feature)
    model = Sequential()
    model.add(Embedding(max_feature, 128))
    print('Embedding_out_shape:', model.output_shape)
    model.add(LSTM(128))
    model.add(Dense(num_features, activation='sigmoid'))
    return model


def train(x_train, y_train, x_test, y_test, max_feature ,  model_dict):
    """
    model_dict = {
        'model_name' : 'nlp',
        'batch_size' : 32,
        'epochs' : 50, 
        'loss': 'categorical_crossentropy'
        'optimizer' :'rmsprop',
        'metrics': ['accuracy']
    }
    """

    model_name = model_dict['model_name']
    batch_size = model_dict['batch_size']
    epochs = model_dict['epochs']
    loss=  model_dict['loss']
    optimizer= model_dict['optimizer']
    metrics = model_dict['metrics']

    model = get_model(max_feature=max_feature, num_features=len(y_train[0]))

    model.compile(loss=loss,
                  optimizer=optimizer, metrics=metrics)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              shuffle=True, validation_data=(x_test, y_test))

    result = model.evaluate(x_test, y_test)

    model.save_weights('./model/weights/' + model_name + '.h5')
    model_json = model.to_json()
    json_name = './model/models/' + model_name + '.json'

    with open(json_name, 'w+') as json_file:
        json_file.write(model_json)
    json_file.close()

    # save meta data for model
    meta = {
        "loss": loss,
        "optimizer": optimizer,
        "metrics": metrics
    }
    meta_string = json.dumps(meta)
    meta_name = './model/meta/' + model_name + '.json'
    with open(meta_name, 'w+') as meta_file:
        meta_file.write(meta_string)
    meta_file.close()

    print("Network trained and saved as %s" % (model_name))

    # clear sessions and graphs
    K.clear_session()
    del model
    return result

def load_k_model(json_path, weight_path, meta_path):
    """ Function to load model using json and weight path
    Arguments
    ---------
    json_path: path to json file
    weight_path: path to weight file
    meta_path: path to meta file
    Returns
    -------
    model: keras model
    """
    K.clear_session()
    # load json file
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # load model
    model = model_from_json(loaded_model_json)
    # load weights
    model.load_weights(weight_path)
    # load metadata
    meta_file = open(meta_path, 'r')
    loaded_model_meta = meta_file.read()
    meta = json.loads(loaded_model_meta)
    meta_file.close()
    # compile model
    model.compile(loss =meta['loss'], optimizer=meta['optimizer'], metrics=meta['metrics'])
    return model

def get_accuracy(y_true, y_pred):
    mlp = MultiLabelBinarizer()
    mlp.fit(y_pred+y_true)
    y_true = mlp.transform([y for y in y_true])
    y_pred = mlp.transform([y for y in y_pred])
    result = {}
    result['precision'] = precision_score(y_true,y_pred,average='micro')
    result['recall'] = recall_score(y_true,y_pred, average='micro')
    result['f1'] = f1_score(y_true,y_pred, average='micro')
    result['accuracy'] = accuracy_score(y_true, y_pred)
    return result

def visualize_model(model, path):
    plot_model(model, to_file=path)



class Preprocess(object):
    def __init__(self, maxlen=200, tokensize=10000):
        self.mlp = MultiLabelBinarizer()
        self.maxlen = maxlen
        self.tokensize = tokensize
        self.tokenizer = Tokenizer(num_words = tokensize)
       
    
    def return_data_labels(self,train_path, test_path, maxlen=200,type='train', ftype='csv'):
        train_data, train_labels = load_data(train_path, ftype=ftype)
        test_data, test_labels = load_data(test_path, ftype=ftype)


        train_data = self.tokenize(data=train_data)
        test_data = self.tokenize(data=test_data)

        train_data = pad_sequences(train_data, maxlen=self.maxlen)
        test_data = pad_sequences(test_data, maxlen=self.maxlen)

        self.mlp.fit(train_labels + test_labels)

        train_labels = self.mlp.transform([ label for label in train_labels])
        test_labels = self.mlp.transform([label for label in test_labels])
        
        if(type=='train'):
            with open('./model/labels/labels.txt','w+') as file:
                for class_name in self.mlp.classes_ :
                    file.write(class_name+ '\n')
                file.close()

        return np.array(train_data),np.array(train_labels), np.array(test_data), np.array(test_labels)
    
    def return_data_csv(self,csv, ftype='csv'):
        data, labels = load_data(csv, ftype=ftype)
        data = self.tokenize(data)
        data = pad_sequences(data, maxlen=self.maxlen)
        return data, labels

    
    def return_data(self, data=[]):
        data = self.tokenize(data=data)
        data = pad_sequences(data, maxlen=self.maxlen)
        return data
 
    def tokenize(self,data=[]):
        self.tokenizer.fit_on_texts(data)
        # return self.tokenizer.texts_to_matrix(data, mode='tfidf')
        return self.tokenizer.texts_to_sequences(data)
    

class detector(object):
    def __init__(self,paths,onlyeone=False):
        if(onlyeone):
            self.model = load_model('./app/model/reuters.h5')
        else:
            self.model = load_k_model(paths['model_path'], paths['weights_path'], paths['meta_path'])
        self.labels = read_file(paths['labels_path'], rtype='list')
        self.labels = [label.strip() for label in self.labels]
        self.graph = tf.get_default_graph()
    
    def predict(self, data):
       with self.graph.as_default():
           result = self.model.predict(data)
           return result
    
    def predict_label(self,data):
        result = self.model.predict_label
        return result

    def get_labels(self, predict, threshold=0.5):
        labels_set = []
        for pred in predict:
            labels = []
            for i in range(len(pred)-1):
                if(pred[i] > threshold):
                    result = {
                        'label' : str(self.labels[i]),
                        'confidence': str(pred[i])
                    }
                    labels.append(result)
            labels_set.append(labels)
        return labels_set
