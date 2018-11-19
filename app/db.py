# # -*- coding: utf-8 -*-
"""
CURL function for MongoDB and FirebaseDataBase
Author: Hein Htet Naing (Hector) , 2018 
"""
import pyrebase

class FireDB(object):
    def __init__(self):
        self.config = {
            "apiKey": "AIzaSyD9_f_1-3v2bXr0BVt4fIj3cuseYIJZE5c",
            "authDomain": "nic-core-2018.firebaseapp.com",
            "databaseURL": "https://nic-core-2018.firebaseio.com",
            "projectId": "nic-core-2018",
            "storageBucket": "nic-core-2018.appspot.com",
            "messagingSenderId": "328640008099"
        }
        self.firebase = pyrebase.initialize_app(self.config)
        self.db = self.firebase.database()

    def insert(self, collection_name, data={},):
        self.db.child(str(collection_name) + '/').set(data)
    
    def push(self, collection_name, data={}):
        self.db.child(collection_name).push(data)

    def get(self, collection_name):
        result = self.db.child(collection_name).get()
        # for r in result.each():
        #     data.append(r.val())
        return result.val()

    def remove(self, collection_name=''):
        self.db.child(collection_name).remove()

