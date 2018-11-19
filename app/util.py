import csv
import nltk
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer



def read_file(file, rtype=' '):
    file = open(file, encoding="utf8", errors='ignore')
    if(rtype=='list'):
            return file.readlines()
    else:
            return file.read()
def read_json(file):
        file=open(file)
        data = json.load(file)
        return data


def read_csv(path):
    data = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            data.append(row)
    data = data[1:]
    data_set = [d[0] for d in data]
    labels = [d[1:] for d in data]
    return data_set, labels


def load_data(path, ftype='csv'):
        if(ftype=='csv'):
                text, labels = read_csv(path)
                data = []
                for t in text:
                        data.append(read_file(t))
                return data, labels
        elif(ftype=='jsonlist'):
                all_json = []
                for p in path:
                        all_json.append(read_json(p))
                data_set = []
                labels =[]
                for json in all_json:
                        for dictionary in json:
                                data_set.append(dictionary['textdata'])
                                labels.append(dictionary['concepts'])
                return data_set, labels
        elif(ftype=='json'):
                data = read_json(path)
                data_set =[]
                labels =[]
                for d in data:
                        data_set.append(d['textdata'])
                        labels.append(d['concepts'])
                print(len(data_set), len(labels))
                return data_set, labels
