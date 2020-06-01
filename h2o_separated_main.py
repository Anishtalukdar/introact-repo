# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:23:00 2020

@author: Datacore
"""


from flask import Flask, request, redirect, url_for, flash, jsonify, Response
from flask_restful import Resource
import logging
import os 
#from google.cloud import storage
#from main_google_model import googleautoml
#added for blueprint
#from main_google_model import googleautoml_api
#from main_azure import azureautoml_api
from main_H2o import h2oautoml_api
#from main_MBA import mbaautoml_api
app = Flask(__name__)

app.register_blueprint(h2oautoml_api)

if __name__ == "__main__":
    #app.run(host='localhost', debug=True, port=5000)
    app.run(host='10.12.1.206', debug=True, port=54321)
    #app.run(host='localhost', debug=True, port=5000)