from datetime import datetime   
from app import NIC_ML
import numpy as np

model_name = 'reuters'
nic_ml = NIC_ML(model_name, True)


bbc = nic_ml.get_data_firedb('/news/bbc')
cna = nic_ml.get_data_firedb('/news/channelnewsasia')
theguardian = nic_ml.get_data_firedb('/news/theguardian')
irishtimes = nic_ml.get_data_firedb('/news/irishtimes')
straitstimes = nic_ml.get_data_firedb('/news/straitstimes')

all_source = [bbc, cna, theguardian, irishtimes, straitstimes]


count = 0
for idx,source in enumerate(all_source):
    
    for news in source:
        
        if(idx >2):
            news = source[news]
            path =  news['url'].replace('/','-')
            path = path.replace('.','-')
     
        if(idx <3):
            path =  news['url'].replace('/','-')
       
        textdata = [news['textdata']]
        concepts = nic_ml.predict(textdata)

        process_data={}
        process_data[path] = {}
        process_data[path]['concpets'] = concepts[0]
        process_data[path]['url'] = 'www.' + news['source'] + '.com' + news['url']
        process_data[path]['published-time'] = news['time']
        
        nic_ml.save_to_firedb(collection_name='nic-concepts', newssite=news['source'] + '/' + path  + '/', data=process_data)

        for concept in concepts[0]:
            label = concept['label']
            pred_concepts = {}
            pred_concepts['url'] = 'www.' + news['source'] + '.com' + news['url']
            pred_concepts['published-time'] = news['time']
    
            nic_ml.save_to_firedb(collection_name='concepts/' + label + '/' , newssite= path , data=pred_concepts)

        count+=1
        
    
logger ={}
logger['nic-ml'] = {

    'total_processed' : count,
    'time': str(datetime.now().isoformat())
}
nic_ml.save_to_firedb('logger', data=logger)
        

    