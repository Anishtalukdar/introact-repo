#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Friday Sep 27 12:01:13 2019

@author: datacore
"""
from flask import Flask, request, redirect, url_for, flash, jsonify, Response
from flask_restful import Resource
import logging
import os 
from google.cloud import storage
from main_google_model import googleautoml
#added for blueprint
from main_google_model import googleautoml_api
from main_azure import azureautoml_api
#from main_H2o import h2oautoml_api
from main_MBA import mbaautoml_api
app = Flask(__name__)

#added for blueprint
app.register_blueprint(googleautoml_api)
app.register_blueprint(azureautoml_api)
#app.register_blueprint(h2oautoml_api)
app.register_blueprint(mbaautoml_api)

class bucket:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "jovial-talon-263619-b9acd4414024.json"
    #logging.basicConfig(filename = "googleautoml.log", level=logging.INFO)
    #logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

    """Creates a new bucket."""
    @app.route('/api/create_bucket', methods=['POST'])
    def create_bucket():

        bucket_name = request.json['bucketname']
        locationType = request.json['locationType']
        storage_class = request.json['storage_class']
        location = request.json['googlelocation']  

        # bucket_name_validation(bucket_name)
               
        try:
            # bucket_name = input("Provide a bucket name to create a new bucket: ")
            storage_client = storage.Client()
            # Create a new bucket object
            bucket = storage_client.bucket(bucket_name)
            # Set Location Type
            # bucket.location_type = "Region"
            # Set location
            # bucket.location = "us-central1"
            # Set storage class
            # bucket.storage_class = "STANDARD"
            
            # Set location
            bucket.location = location
            # Set storage class
            bucket.storage_class = storage_class
            # Create bucket 
            bucket.create() 
            logging.info('Bucket {} created'.format(bucket.name))
            return 'Bucket {} created successfully'.format(bucket.name)
        except Exception as e:
            if "https" in str(e):
                error_statement = str(e).partition(": ")[2]            
            else:
                error_statement = str(e)
            print("Error statement: ",error_statement)
            return error_statement

    
    """Deletes a bucket. The bucket must be empty."""
    @app.route('/api/del_bucket', methods=['POST'])  
    def delete_bucket():    
        # bucket_name = input("Provide a bucket name to delete the bucket: ")
        bucket_name = request.json['bucketname']
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        bucket.delete()
        logging.info('Bucket {} deleted'.format(bucket.name))
        return 'Bucket {} deleted successfully'.format(bucket.name)
        
    @app.route('/api/bucket_exist', methods=['POST'])
    def bucket_exist():
        bucket_name = request.json['bucketname']
        locationType = request.json['locationType']
        storage_class = request.json['storage_class']
        location = request.json['googlelocation']
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            bucket.location = location
            bucket.storage_class = storage_class

            if bucket == storage_client.get_bucket(bucket_name):
                print('Bucket {} already exist'.format(bucket_name))
            return "Bucket Exist"
        except Exception as e:
            error_statement = str(e)
            print("Error statement: ",error_statement)
            print("Bucket does not exist, need to create new bucket")
            #return "Bucket does not exist" ****need to check
            return "Bucket Does Not Exist"

    """Uploads a file to the bucket and import dataset."""
    @app.route('/api/upload_blob_importDataset', methods=['POST'])   
    def upload_blob():

        bucket_name = request.json['bucketname']
        destination_blob_name = request.json['des_file_name']
        source_file_name = request.json['src_file_name']    
        dataset_display_name = request.json['dataset_display_name']    

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
            
        try:
            blob.upload_from_filename(source_file_name)
            logging.info('File {} uploaded to {}.'.format(
            source_file_name,
            destination_blob_name))
            #print('File {} successfully uploaded to {}.'.format(destination_blob_name,bucket_name))
            googleautoml.importDataset(bucket_name, destination_blob_name, dataset_display_name)
            return "File uploaded in the bucket and dataset imported successfully" 
        except Exception as e: 
            if "https" in str(e):
                error_statement = str(e).partition(": ")[2]            
            else:
                error_statement = str(e)
            logging.error(str(e))            
            return error_statement
    

    """Uploads a file to the bucket."""
    # @app.route('/api/upload_blob_prediction_file', methods=['POST'])   
    # def upload_prediction_blob():    
    def upload_prediction_blob(bucket_name, predict_gcs_input_uris, source_file_name):

        # bucket_name = request.json['bucketname']
        # destination_blob_name = request.json['des_file_name']
        # source_file_name = request.json['src_file_name'] 
        
        destination_blob_name = predict_gcs_input_uris        

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
            
        try:
            blob.upload_from_filename(source_file_name)
            logging.info('File {} uploaded to {}.'.format(
            source_file_name,
            destination_blob_name))
            print('Prediction File {} successfully uploaded to {}.'.format(destination_blob_name,bucket_name))            
            #return "Prediction file successfully uploaded to bucket" 
        except Exception as e: 
            if "https" in str(e):
                error_statement = str(e).partition(": ")[2]            
            else:
                error_statement = str(e)
            logging.error(str(e))
            print("Error statement: ",error_statement)
            #return error_statement
    

    """Downloads a blob from the bucket."""
    @app.route('/api/download_blob', methods=['POST'])
    def download_blob():   
        bucket_name = request.json['bucketname']
        destination_file_name = request.json['des_file_name']
        source_blob_name = request.json['src_file_name']

        storage_client = storage.Client()
        # bucket_name = input("Enter the bucket name you want the file to be downloaded from: ")
        bucket = storage_client.get_bucket(bucket_name)
        # source_blob_name = input("Enter the source blob name: ")
        blob = bucket.blob(source_blob_name)

        # destination_file_name = input("Enter the destination file name: ")
        blob.download_to_filename(destination_file_name)
        logging.info('Blob {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))
        return 'File downloaded properly'


    # """Deletes a blob from the bucket."""
    @app.route('/api/delt_blob', methods=['POST'])
    def delete_blob():       
        bucket_name = request.json['bucketname']
        blob_name = request.json['blob_name']
        
        storage_client = storage.Client()
        #bucket_name = input("Enter the bucket name from which you want to delete: ")
        bucket = storage_client.get_bucket(bucket_name)
        #blob_name = input("Enter the blob name you want to delete: ")
        blob = bucket.blob(blob_name)
        blob.delete()        
        logging.info('Blob {} deleted.'.format(blob_name))
        return 'ok'

    """Lists all the blobs in the bucket."""
    @app.route('/api/list_blob', methods=['POST'])
    def list_of_blob():
        bucket_name = request.json['bucketname']
        storage_client = storage.Client()
        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(bucket_name)
        for blob in blobs:
            print("Blob names: ",blob.name)
        return 'ok'       


    @app.route('/api/download_model', methods=['POST'])
    def model_download():        
        """Downloads a blob from the bucket."""
        bucket_name = request.json["bucket_name"]        
        source_blob_name =  request.json["source_blob_name"]        
        destination_file_name = request.json["destination_file_name"]        
        storage_client = storage.Client()
        
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        print("blob: ",blob)

        
        blob.download_to_filename(destination_file_name)

        print('Blob {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))

        return "Model downloaded successfully" 
        

if __name__ == "__main__":
    #app.run(host='localhost', debug=True, port=5000)
    app.run(host='10.12.1.206', debug=True, port=5000)
    #app.run(host='localhost', debug=True, port=5000)
    
    
    

