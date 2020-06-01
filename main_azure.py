# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:01:13 2019

@author: datacore
"""

from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_restful import Resource
import pandas as pd
import os
import json
from azureml.core.workspace import Workspace
import azureml.core
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azure.storage.blob import BlockBlobService
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import Model
#from azureml.train.automl.runtime import AutoMLStep



#import logging

from flask import Blueprint
import zipfile

print(azureml.core.VERSION)
#added for blueprint
azureautoml_api = Blueprint('azureautoml_api', __name__)

#app = Flask(__name__)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
@azureautoml_api.route('/api/WS_create', methods=['POST'])
def WSCreate():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        
       ##Existing \ new workspace
        try:
            ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                  
            
            print("Found workspace {} at location {}".format(ws.name, ws.location))
            print('Found existing Workspace.')
            return "Workspace exist"
        except:
            print('need to create new Workspace.')
            print('Creating new Workspace.')   
            ws = Workspace.create(name=workspace_name,
                               subscription_id=subscription_id,
                               resource_group=resource_group,
                               #create_resource_group=True,
                               location=location)
            return "Workspace successfully created"

@azureautoml_api.route('/api/WS_exist', methods=['POST'])
def WSExist():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        
       ##Existing \ new workspace
        try:
            ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                  
            
            print("Found workspace {} at location {}".format(ws.name, ws.location))
            print('Found existing Workspace.')
            return "Workspace exist"
        except:
            print('need to create new Workspace.')
            return "Workspace not exist"
        
@azureautoml_api.route('/api/ws_delete', methods=['POST'])
def WSDelete():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        try:
            ws.delete(delete_dependent_resources=True, no_wait=False)
            print('Workspace deleted')
            return "Workspace deleted"
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            return error_statement
            
      
@azureautoml_api.route('/api/compute_create', methods=['POST'])
def ComputeCompute():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        cluster_name = request.json['cluster_name']
        vm_size = request.json['vm_size']
        min_nodes = request.json['min_nodes']
        max_nodes = request.json['max_nodes']
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                  
            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        #aml_compute = AmlCompute(ws, cluster_name)
        #cluster_name = 'cpu-cluster'
        
        try:
            aml_compute = AmlCompute(ws, cluster_name)
            print('Found existing AML compute context.')
            return "Found existing AML compute context."
        except:
            print('need to create new Compute.')
            print('Creating new AML compute context.')
            aml_config = AmlCompute.provisioning_configuration(vm_size = vm_size, min_nodes=min_nodes, max_nodes=max_nodes)
            aml_compute = AmlCompute.create(ws, name = cluster_name, provisioning_configuration = aml_config)
            aml_compute.wait_for_completion(show_output = True)
            return "Compute successfully created"
        
@azureautoml_api.route('/api/compute_exist', methods=['POST'])
def ComputeExist():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        Cluster_type = request.json['Cluster_type']
        cluster_name = request.json['cluster_name']
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                  
            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        #aml_compute = AmlCompute(ws, cluster_name)
        #cluster_name = 'cpu-cluster'
        try:
            if Cluster_type == 'Training':
                aml_compute = AmlCompute(ws, cluster_name)
            else: 
                aks_target = AksCompute(ws,cluster_name)
            print('Found existing AML compute context.')
            return "compute exist"
        except:
            print('need to create new Compute.')
            return "compute not exist"   
                  
@azureautoml_api.route('/api/compute_delete', methods=['POST'])
def ComputeDelete():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        Cluster_type = request.json['Cluster_type']
        cluster_name = request.json['cluster_name']
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        try:
            if Cluster_type == 'Training':
                aml_compute = AmlCompute(ws, cluster_name)
                print('Found existing AML compute context.')
                aml_compute.delete()
            else:            
                aks_target = AksCompute(ws,cluster_name)
                print('Found existing AKS compute context.')
                aks_target.delete()           
            print('compute deleted')
            return "compute deleted"
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            return error_statement
  

@azureautoml_api.route('/api/aks_create', methods=['POST'])
def AKSCompute():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        cluster_name = request.json['cluster_name']
        vm_size = request.json['vm_size']
        agent_count = request.json['agent_count']
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                  
            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        #aml_compute = AmlCompute(ws, cluster_name)
        #cluster_name = 'cpu-cluster'
        try:
            aks_target = AksCompute(ws,cluster_name)
            print('Found existing AKS compute context.')
            return "Found existing AKS compute context."
        except:
            print('need to create new Compute.')
            print('Creating new AKS compute context.')
            prov_config = AksCompute.provisioning_configuration(vm_size = vm_size,
                                                       agent_count = agent_count,
                                                       location = location)
            aks_target = ComputeTarget.create(workspace = ws,
                                              name = cluster_name,
                                              provisioning_configuration = prov_config)

            # Wait for the create process to complete
            aks_target.wait_for_completion(show_output = True)
            return "Compute successfully created"
            
@azureautoml_api.route('/api/upload_data', methods=['POST'])
def UploadCSV():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
    
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        
        ds = ws.get_default_datastore()
        print(ds.datastore_type, ds.account_name, ds.container_name)
        file_path = request.json['file_path']
        print(file_path)
        file_name = request.json['file_name']
        ds.upload(src_dir=file_path, target_path= None, overwrite=True, show_progress=True)
        try:
            stock_ds = Dataset.Tabular.from_delimited_files(path=ds.path(file_name))
            stock_ds = stock_ds.register(workspace = ws,
                                     name = file_name,
                                     description = 'stock training data',create_new_version=True)
            print('Data Registered to the ML Workspace.')
            return "Data Registered to the ML Workspace."
        except Exception as e:
            error_statement = str(e)
            print("Error statement: ",error_statement)
            print('dataset is not Registered, please check')
        return "Dataset is not Registered, please check"

@azureautoml_api.route('/api/register_data_from_blob', methods=['POST'])
def RegisterCSV():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
    
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        
        ds = ws.get_default_datastore()
        #print(ds.datastore_type, ds.account_name, ds.container_name)
        #file_path = request.json['file_path']
        #print(file_path)
        file_name = request.json['file_name']
        #ds.upload(src_dir=file_path, target_path= None, overwrite=True, show_progress=True)
        try:
            stock_ds = Dataset.Tabular.from_delimited_files(path=ds.path(file_name))
            stock_ds = stock_ds.register(workspace = ws,
                                     name = file_name,
                                     description = 'stock training data',create_new_version=True)
            print('Data Registered to the ML Workspace.')
            return "Data Registered to the ML Workspace."
        except:
            print('dataset is not Registered, please check')
        return "Dataset is not Registered, please check"

    
@azureautoml_api.route('/api/blob_data', methods=['POST'])
def BlobData():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
    
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        
        ds = ws.get_default_datastore()
        print(ds.datastore_type, ds.account_name, ds.container_name)
        block_blob_service = BlockBlobService(account_name=ds.account_name, account_key=ds.account_key)
        try:
            blobs = []
            my_list=[]
            marker = None
            while True:
                batch = block_blob_service.list_blobs(ds.container_name,prefix = 'H')
                blobs.extend(batch)
                if not batch.next_marker:
                    break
                marker = batch.next_marker
            for blob in blobs:
                print(blob.name)
                my_list.append(blob.name)
            print(my_list)
            my_json_string = json.dumps(my_list)
            print('dataset is fetched from blob, please check')
            return my_json_string
        except:
            print('dataset is not fetched from blob, please check')
        return "Dataset is not fetched from blob, please check"   
    
@azureautoml_api.route('/api/blob_data_download', methods=['POST'])
def BlobDataDownload():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        file_name = request.json['file_name']

    
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        
        ds = ws.get_default_datastore()
        print(ds.datastore_type, ds.account_name, ds.container_name)
        block_blob_service = BlockBlobService(account_name=ds.account_name, account_key=ds.account_key)
        #file_name = 'RetailChurnTemplate_FeatureEngg_ProcessedData_20.csv'
        try:
            local_path='D:\\DCSAIAUTOML\\TempFolder'
            full_path_to_file =os.path.join(local_path, file_name)
            print(full_path_to_file)
            block_blob_service.get_blob_to_path(ds.container_name, file_name, full_path_to_file,start_range=0,end_range=1100)
            df = pd.read_csv(full_path_to_file) 
            df.head(100)
            dfff_json = df.to_json(orient='records')
            #print(dfff_json)
            return dfff_json
        except:
            print('dataset is not saved from blob, please check')
        return "Dataset is not saved from blob, please check" 
    
@azureautoml_api.route('/api/AutoMLRun_Azure_class', methods=['POST'])
def RunAutoMLClass():
        
        
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        file_name = request.json['file_name']
        location = request.json['location']
        target_var = request.json['target_var']
        cluster_name = request.json['cluster_name']
        best_model = request.json['best_model']
        #best_model = request.json['best_model']
    
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        #compute_target = AmlCompute(ws, cluster_name)
        compute_target = ws.compute_targets[cluster_name]
        print('Found existing AML compute context.')    
        dataset_name = file_name

        # Get a dataset by name
        df = Dataset.get_by_name(workspace=ws, name=dataset_name)
        #stock_dataset_df = df.to_pandas_dataframe()
        print('file successfully recieved.')
        #stock_dataset_df.head()
        #stock_dataset_json = stock_dataset_df.to_json(orient='split')
        #print(stock_dataset_json)
        X = df.drop_columns(columns=[target_var])
        y = df.keep_columns(columns=[target_var], validate=True)
        #y_df = stock_dataset_df[target_var].values
        #x_df = stock_dataset_df.drop([target_var], axis=1)
        print(y)
        # create a new RunConfig object
        conda_run_config = RunConfiguration(framework="python")        
        conda_run_config.environment.docker.enabled = True
        conda_run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_CPU_IMAGE        
        cd = CondaDependencies.create(pip_packages=['azureml-sdk[automl]'],
                                      conda_packages=['numpy', 'py-xgboost<=0.90'])
        conda_run_config.environment.python.conda_dependencies = cd       
        print('run config is ready')
        ExperimentName = request.json['ExperimentName']       
        tasks = request.json['tasks']
        iterations = request.json['iterations']
        n_cross_validations = request.json['n_cross_validations']
        iteration_timeout_minutes = request.json['iteration_timeout_minutes']
        primary_metric = request.json['primary_metric']
        max_concurrent_iterations = request.json['max_concurrent_iterations']       
        
        try:
            automl_settings = {
                "name": ExperimentName,
                "iteration_timeout_minutes": iteration_timeout_minutes,
                "featurization": 'auto',
                "iterations": iterations,
                "n_cross_validations": n_cross_validations,
                "primary_metric": primary_metric,
                "preprocess": True,
                "max_concurrent_iterations": max_concurrent_iterations
                #"verbosity": logging.INFO
            }

            automl_config = AutoMLConfig(task=tasks,
                                         debug_log='automl_errors.log',
                                         blacklist_models =['XGBoost'],
                                         #path=os.getcwd(),
                                         compute_target=compute_target,
                                         #run_configuration=conda_run_config,
                                         X=X,
                                         y=y,
                                         **automl_settings,
                                        )

            experiment=Experiment(ws, ExperimentName)
            remote_run = experiment.submit(automl_config, show_output=True)
            remote_run.flush(timeout_seconds=3600)            
            children = list(remote_run.get_children())
            metricslist = {}
            for run in children:
                properties = run.get_properties()
                metrics = {k: v for k, v in run.get_metrics().items() if isinstance(v, float)}
                metricslist[int(properties['iteration'])] = metrics
            
            algo_list=[]
            for child_run in remote_run.get_children():
                properties = child_run.get_properties()
                algo_name = properties.get('run_algorithm')
                print(algo_name)
                algo_list.append(algo_name)
                #print(lalgo_list)
                #rundata1 = pd.DataFrame(algo_list)
                #rundata1.rename(columns={'0':'algirithm'})
                print(algo_list)
                
            rundata = pd.DataFrame(metricslist).sort_index(axis=1, by= primary_metric)
            rundata = rundata.drop(['AUC_macro', 'AUC_micro','average_precision_score_macro','average_precision_score_micro', 'average_precision_score_weighted', 
              'balanced_accuracy','f1_score_macro', 'f1_score_micro','norm_macro_recall', 'precision_score_macro', 'precision_score_micro', 
              'recall_score_macro', 'recall_score_micro', 'weighted_accuracy' ])
            rundata.rename(columns = {0: "one", 1: "two",2: "three",3: "four",4: "five",5: "six",6: "seven",
                                    7: "eight",8: "nine",9: "ten",}, inplace = True)  
            iterations_toJson = rundata.to_json(orient='columns')
            print(iterations_toJson)
            #run_details = remote_run.get_details()            
            #print(run_details)
            best_run, fitted_model = remote_run.get_output()
            best_run_toJson = best_run.get_metrics()
            cwd = 'D:\\DCSAIAUTOML\\BestModels\\Azure\\'
            best_model_name = best_run.name
            model = remote_run.register_model(description = best_model)
            print(model.name, model.id, model.version, sep = '\t')
            import os
            model_path = os.path.join(cwd, best_model, best_model_name)
            print(model_path)
               
            
            #print("Model DownLoad Complete")
            #model = Model(workspace=ws, name=model.name)
            #model.download_files(target_dir=model_path)
            #dict = {}
            #dict['iterations_toJson'] = iterations_toJson
            #dict['best_run_toJson'] = best_run_toJson
            #print(best_run.get_file_names())
            #Register the model
            #from datetime import date

            best_model_id = best_run.name

            var1 = "@"
            var2 = var1 + best_model_id

            Reg_model_name = model.name
            var4  = var1 + Reg_model_name

            best_run.flush(timeout_seconds=3600)
            best_run.download_files(output_directory = model_path) 
            # importing required modules        
            #import shutil
            #output_path = os.path.join(model_path, best_model_id)
            #dir_name1 = "D:\\DCSAIAUTOML\\BestModels\\Azure\\my_azure_best"
            #dir_name1 = "D:\\DCSAIAUTOML\\BestModels\\Azure\\my_azure_best\\my_azure_best"
            #shutil.make_archive(model_path,'zip',model_path)
            
            #zipf = zipfile.ZipFile(best_model_id+'.zip', 'w', zipfile.ZIP_DEFLATED)
            #for root, dirs, files in os.walk(model_path):
                #for file in files:
                    #zipf.write(os.path.join(root, file))
                    
            #def zipdir(path, ziph):
                # ziph is zipfile handle
                #import os
                #for root, dirs, files in os.walk(path):
                    #for file in files:
                        #ziph.write(os.path.join(root, file))
           
            #zipdir(model_path, zipf)
            #remote_run.clean_preprocessor_cache()
            print("ready to return")
            var5 = "no exception"
            return  '{} {} {} {} {} {} {}'.format(iterations_toJson, var2,var4,var1,var5,var1,algo_list)              
            #return iterations_toJson
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            model_path1 = os.path.join(model_path,'outputs')
            file_name = 'model.pkl'
            print("in exception: " ,model_path1)            
            src = 'D:\\Final Script_dev'
            full_file_name = os.path.join(src, file_name)
            import shutil
            #remote_run.download_file('model.pkl', output_file_path=model_path1)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, model_path1)
            return  '{} {} {} {} {}'.format(iterations_toJson, var2, var4,var1,error_statement)
        
@azureautoml_api.route('/api/AutoMLRun_Azure_reg', methods=['POST'])
def RunAutoMLReg():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        file_name = request.json['file_name']
        location = request.json['location']
        target_var = request.json['target_var']
        cluster_name = request.json['cluster_name']
        best_model = request.json['best_model']
        #best_model = request.json['best_model']
    
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        #compute_target = AmlCompute(ws, cluster_name)
        compute_target = ws.compute_targets[cluster_name]
        print('Found existing AML compute context.')    
        dataset_name = file_name

        # Get a dataset by name
        df = Dataset.get_by_name(workspace=ws, name=dataset_name)
        #stock_dataset_df = df.to_pandas_dataframe()
        print('file successfully recieved.')
        #stock_dataset_df.head()
        #stock_dataset_json = stock_dataset_df.to_json(orient='split')
        #print(stock_dataset_json)
        X = df.drop_columns(columns=[target_var])
        y = df.keep_columns(columns=[target_var], validate=True)
        #y_df = stock_dataset_df[target_var].values
        #x_df = stock_dataset_df.drop([target_var], axis=1)
        print(y)
        # create a new RunConfig object
        conda_run_config = RunConfiguration(framework="python")        
        conda_run_config.environment.docker.enabled = True
        conda_run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_CPU_IMAGE        
        cd = CondaDependencies.create(pip_packages=['azureml-sdk[automl]'], 
                                      conda_packages=['numpy', 'py-xgboost<=0.90'])
        conda_run_config.environment.python.conda_dependencies = cd       
        print('run config is ready')
        ExperimentName = request.json['ExperimentName']       
        tasks = request.json['tasks']
        iterations = request.json['iterations']
        n_cross_validations = request.json['n_cross_validations']
        iteration_timeout_minutes = request.json['iteration_timeout_minutes']
        primary_metric = request.json['primary_metric']
        max_concurrent_iterations = request.json['max_concurrent_iterations']       
        
        try:
            automl_settings = {
                "name": ExperimentName,
                "iteration_timeout_minutes": iteration_timeout_minutes,
                "featurization": 'auto',
                "iterations": iterations,
                "n_cross_validations": n_cross_validations,
                "primary_metric": primary_metric,
                "preprocess": True,
                "max_concurrent_iterations": max_concurrent_iterations
                #"verbosity": logging.INFO
            }

            automl_config = AutoMLConfig(task=tasks,
                                         debug_log='automl_errors.log',
                                         blacklist_models =['XGBoost'],
                                         #path=os.getcwd(),
                                         compute_target=compute_target,
                                         #run_configuration=conda_run_config,
                                         X=X,
                                         y=y,
                                         **automl_settings,
                                        )

            experiment=Experiment(ws, ExperimentName)
            remote_run = experiment.submit(automl_config, show_output=True)  
            remote_run.flush(timeout_seconds=400)            
            children = list(remote_run.get_children())
            metricslist = {}
            for run in children:
                properties = run.get_properties()
                metrics = {k: v for k, v in run.get_metrics().items() if isinstance(v, float)}
                metricslist[int(properties['iteration'])] = metrics
            
            rundata = pd.DataFrame(metricslist).sort_index(axis=1, by= primary_metric)
            rundata = rundata.drop(['mean_absolute_percentage_error', 'normalized_median_absolute_error','normalized_root_mean_squared_log_error','root_mean_squared_log_error'])
            rundata.rename(columns = {0: "one", 1: "two",2: "three",3: "four",4: "five",5: "six",6: "seven",
                                    7: "eight",8: "nine",9: "ten",}, inplace = True)  
            iterations_toJson = rundata.to_json(orient='columns')
            print(iterations_toJson)
            best_run, fitted_model = remote_run.get_output()
            best_run_toJson = best_run.get_metrics()
            cwd = 'D:/DCSAIAUTOML/BestModels/Azure'
            best_model_name = best_run.name
            model = remote_run.register_model(description = best_model)
            print(model.name, model.id, model.version, sep = '\t')
            model_path = os.path.join(cwd, best_model, best_model_name)
            print(model_path)
            #print("Model DownLoad Complete")
            #model = Model(workspace=ws, name=model.name)
            #model.download_files(target_dir=model_path)
            #dict = {}
            #dict['iterations_toJson'] = iterations_toJson
            #dict['best_run_toJson'] = best_run_toJson
            #print(best_run.get_file_names())
            #Register the model
            #from datetime import date

            best_model_id = best_run.name

            var1 = "@"
            var2 = var1 + best_model_id

            Reg_model_name = model.name
            var4  = var1 + Reg_model_name

            best_run.flush(timeout_seconds=3600)
            best_run.download_files(output_directory = model_path) 
            # importing required modules        
            #import shutil
            #output_path = os.path.join(model_path, best_model_id)
            #dir_name1 = "D:\\DCSAIAUTOML\\BestModels\\Azure\\my_azure_best"
            #dir_name1 = "D:\\DCSAIAUTOML\\BestModels\\Azure\\my_azure_best\\my_azure_best"
            #shutil.make_archive(model_path,'zip',model_path)
            
            #zipf = zipfile.ZipFile(best_model_id+'.zip', 'w', zipfile.ZIP_DEFLATED)
            #for root, dirs, files in os.walk(model_path):
                #for file in files:
                    #zipf.write(os.path.join(root, file))
                    
            #def zipdir(path, ziph):
                # ziph is zipfile handle
                #import os
                #for root, dirs, files in os.walk(path):
                    #for file in files:
                        #ziph.write(os.path.join(root, file))
           
            #zipdir(model_path, zipf)
            #remote_run.clean_preprocessor_cache()
            print("ready to return")
            var5 = "no exception"
            return  '{} {} {} {} {}'.format(iterations_toJson, var2, var4,var1,var5)              
            #return iterations_toJson
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            model_path1 = os.path.join(model_path,'outputs')
            file_name = 'model.pkl'
            print("in exception: " ,model_path1)            
            src = 'D:\\Final Script_dev'
            full_file_name = os.path.join(src, file_name)
            import shutil
            #remote_run.download_file('model.pkl', output_file_path=model_path1)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, model_path1)
            return  '{} {} {} {} {}'.format(iterations_toJson, var2, var4,var1,error_statement)
        
@azureautoml_api.route('/api/deploy_azure_aks', methods=['POST'])
def DeployAzureAKS():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        best_model = request.json['best_model']
        Model_path = request.json['Model_path']
        cluster_name = request.json['cluster_name']
        service_name = request.json['service_name']
        Reg_model_name = request.json['Reg_model_name']
    
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        
        from azureml.core.model import Model
        model = Model(ws, name=Reg_model_name)
        print(model)
        
        from azureml.core.model import InferenceConfig
        from azureml.core.webservice import AciWebservice
        from azureml.core.webservice import Webservice
        from azureml.core.model import Model
        from azureml.core.environment import Environment
        
        from sklearn.externals import joblib
        cwd = 'D:\\DCSAIAUTOML\\BestModels\\Azure'
        model_path = os.path.join(cwd, Model_path, best_model, "outputs")
        #model_path1 = os.path.join(model_path, "outputs", "model.pkl")
        print(model_path)
        os.chdir(model_path)
        #import importlib
        script_file_name = 'scoring_file_v_1_0_0.py'
        conda_env_file_name = 'conda_env_v_1_0_0.yml'
        #importlib.import_module('scoring_file_v_1_0_0.py')
        #script_file_name = joblib.load('scoring_file_v_1_0_0.py')
        #import yaml
        #conda_env_file_name = yaml.load(open('conda_env_v_1_0_0.yml'))
        #conda_env_file_name = joblib.load('conda_env_v_1_0_0.yml')
        
        myenv = Environment.from_conda_specification(name="myenv", file_path=conda_env_file_name)
        inference_config = InferenceConfig(entry_script=script_file_name, environment=myenv)
        
        aks_target = AksCompute(ws,cluster_name)
        # If deploying to a cluster configured for dev/test, ensure that it was created with enough
        # cores and memory to handle this deployment configuration. Note that memory is also used by
        # things such as dependencies and AML components.
        try:
            deployment_config = AksWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 16,enable_app_insights=True,collect_model_data=True,)
            service = Model.deploy(ws, service_name, [model], inference_config, deployment_config, aks_target)
            service.wait_for_deployment(show_output = True)
            print(service.state)
            compute_type = service.compute_type
            state  = service.state
            url = service.scoring_uri
            s_url = service.swagger_uri
            #created_time = service.created_time
            #updated_time = service.updated_time
            v1 = "@"
            v2 = "Deployed Successfully"
            print(v2)
            return '{} {} {} {} {} {} {} {} {}'.format(v2, v1,compute_type,v1,state,v1,url,v1,s_url)
        
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            return error_statement
        
@azureautoml_api.route('/api/deploy_azure_aci', methods=['POST'])
def DeployAzureACI():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        best_model = request.json['best_model']
        Model_path = request.json['Model_path']
        #cluster_name = request.json['cluster_name']
        service_name = request.json['service_name']
        Reg_model_name = request.json['Reg_model_name']
    
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        
        from azureml.core.model import Model
        model = Model(ws, name=Reg_model_name)
        print(model)
        
        from azureml.core.model import InferenceConfig
        from azureml.core.webservice import AciWebservice
        from azureml.core.webservice import Webservice
        from azureml.core.model import Model
        from azureml.core.environment import Environment

        cwd = 'D:\\DCSAIAUTOML\\BestModels\\Azure'
        model_path = os.path.join(cwd, Model_path, best_model, "outputs")
        #model_path1 = os.path.join(model_path, "outputs", "model.pkl")
        print(model_path)
        os.chdir(model_path)
        #import importlib
        script_file_name = 'scoring_file_v_1_0_0.py'
        conda_env_file_name = 'conda_env_v_1_0_0.yml'
        #importlib.import_module('scoring_file_v_1_0_0.py')
        #script_file_name = joblib.load('scoring_file_v_1_0_0.py')
        #import yaml
        #conda_env_file_name = yaml.load(open('conda_env_v_1_0_0.yml'))
        #conda_env_file_name = joblib.load('conda_env_v_1_0_0.yml')
        
        myenv = Environment.from_conda_specification(name="myenv", file_path=conda_env_file_name)
        inference_config = InferenceConfig(entry_script=script_file_name, environment=myenv)

        try:
            deployment_config = AciWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 8)
            service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)
            service.wait_for_deployment(show_output = True)
            print(service.state)
            compute_type = service.compute_type
            state  = service.state
            url = service.scoring_uri
            s_url = service.swagger_uri
            #created_time = service.created_time
            #updated_time = service.updated_time
            v1 = "@"
            v2 = "Deployed Successfully"
            print(v2)
            return '{} {} {} {} {} {} {} {} {}'.format(v2, v1,compute_type,v1,state,v1,url,v1,s_url)
        
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            return error_statement        
        
@azureautoml_api.route('/api/AutoMLRun_Azure_forecast', methods=['POST'])
def RunAutoMLForecast():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        file_name = request.json['file_name']
        location = request.json['location']
        target_var = request.json['target_var']
        cluster_name = request.json['cluster_name']
        best_model = request.json['best_model']
        time_column_name = request.json['time_column_name']
        max_horizon = request.json['max_horizon']
    
        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        compute_target = AmlCompute(ws, cluster_name)
        print('Found existing AML compute context.')    
        dataset_name = file_name
        time_column_name = time_column_name
        # Get a dataset by name
        dataset = Dataset.get_by_name(workspace=ws, name=dataset_name).with_timestamp_columns(fine_grain_timestamp=time_column_name) 
        print(dataset)
        #df_ts = Dataset.Tabular.from_delimited_files(df_ts)
        dataset.to_pandas_dataframe().describe()
        dataset.take(3).to_pandas_dataframe()
        print(dataset)
        #y_df = df_ts[target_var].values
        #x_df = df_ts.drop([target_var], axis=1)
        print('file successfully recieved.')
        #stock_dataset_df.head()
        # create a new RunConfig object
        conda_run_config = RunConfiguration(framework="python")        
        conda_run_config.environment.docker.enabled = True
        conda_run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_CPU_IMAGE        
        cd = CondaDependencies.create(pip_packages=['azureml-sdk[automl]'], 
                                      conda_packages=['numpy', 'py-xgboost<=0.80'])
        conda_run_config.environment.python.conda_dependencies = cd       
        print('run config is ready')
        ExperimentName = request.json['ExperimentName']       
        tasks = request.json['tasks']
        iterations = request.json['iterations']
        n_cross_validations = request.json['n_cross_validations']
        iteration_timeout_minutes = request.json['iteration_timeout_minutes']
        primary_metric = request.json['primary_metric']
        #max_concurrent_iterations = request.json['max_concurrent_iterations']       
        
        automl_settings = {
                'time_column_name': time_column_name,
                'max_horizon': max_horizon,
                "iterations": iterations,
                    }

        automl_config = AutoMLConfig(task=tasks,                             
                             primary_metric=primary_metric,
                             #blacklist_models = ['ExtremeRandomTrees', 'AutoArima', 'Prophet'],                             
                             experiment_timeout_minutes=iteration_timeout_minutes,
                             training_data=dataset,
                             label_column_name=target_var,
                             compute_target=compute_target,
                             enable_early_stopping = True,
                             n_cross_validations=n_cross_validations,                             
                             #verbosity=logging.INFO,
                            **automl_settings)
        print("AutoML config created.")
        experiment=Experiment(ws, ExperimentName)
        remote_run = experiment.submit(automl_config, show_output=True)       
        children = list(remote_run.get_children())
        metricslist = {}
        for run in children:
                    properties = run.get_properties()
                    metrics = {k: v for k, v in run.get_metrics().items() if isinstance(v, float)}
                    metricslist[int(properties['iteration'])] = metrics
                
        rundata = pd.DataFrame(metricslist).sort_index(axis=1, by= primary_metric)
        rundata.rename(columns = {0: "one", 1: "two",2: "three",3: "four",4: "five",5: "six",6: "seven",
                                        7: "eight",8: "nine",9: "ten",}, inplace = True) 
        iterations_toJson = rundata.to_json(orient='columns')
        print(iterations_toJson)
        best_run, fitted_model = remote_run.get_output()
                #best_run_toJson = best_run.get_metrics()
                #dict = {}
                #dict['iterations_toJson'] = iterations_toJson
                #dict['best_run_toJson'] = best_run_toJson
                #print(best_run.get_file_names())
                #Register the model
                #from datetime import date
        model = remote_run.register_model(model_name=best_model, description = 'AutoML Model')
        print(model.name, model.id, model.version, sep = '\t')
        best_model = model.name
        best_model
        var1 = "@"
        var2 = var1 + best_model
        return  '{} {}'.format(iterations_toJson, var2)
                


@azureautoml_api.route('/api/predict_azure', methods=['POST'])
def Prediction():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        file_name = request.json['file_name']
        target_var = request.json['target_var']
        best_model = request.json['best_model']
        Model_path = request.json['Model_path']

        ws = Workspace(subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  workspace_name=workspace_name)
                                  
            
        print("Found workspace {} at location {}".format(ws.name, ws.location))
        print('Found existing Workspace.')
        
        dataset_name = file_name
        # Get a dataset by name
        df = Dataset.get_by_name(workspace=ws, name=dataset_name)
        stock_dataset_df = df.to_pandas_dataframe()
        print('file successfully recieved.')
        X = df.drop_columns(columns=[target_var])
        y = df.keep_columns(columns=[target_var], validate=True)
        y_df = stock_dataset_df[target_var].values
        x_df = stock_dataset_df.drop([target_var], axis=1)
        print(y)
        
        #from azureml.core import Run
        #experiment=Experiment(ws, workspace_name)
        #from azureml.core.model import Model
        #model = Model(ws, name=Model_path)
        #model.download(exist_ok=True)
        from sklearn.externals import joblib
        cwd = 'D:\DCSAIAUTOML\BestModels\Azure'
        model_path = os.path.join(cwd, Model_path, best_model, "outputs")
        #model_path1 = os.path.join(model_path, "outputs", "model.pkl")
        print(model_path)
        os.chdir(model_path)
        model = joblib.load('model.pkl')
        #best_run = Run(experiment=experiment, run_id='AutoML_74e9d9dc-f347-4392-b8bb-3edeb4a6afad_8')
        #fitted_model = Run(experiment=experiment, run_id='AutoML_74e9d9dc-f347-4392-b8bb-3edeb4a6afad_8')
        print(model)
        try:
            y_predict = model.predict(x_df)
            print(y_predict)
            #prediction_toJson = y_predict.to_json(orient='columns')
            #print(prediction_toJson)
            df = pd.DataFrame(y_predict)
            df.rename(columns = {0: "Prediction"}, inplace = True) 
            #stock_df = stock_dataset_df[['SepalLengthCm','SepalWidthCm','Species']]
            result = pd.concat([stock_dataset_df,df],axis=1)
            result.to_csv('D:\\PredictionResult\\Azure\\prediction_azure_health.csv', index=False, date_format='%Y%m%d')
            result.head()
            prediction_toJson = result.to_json(orient='records')
            return prediction_toJson
        
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            return error_statement 
            
        
if __name__ == '__main__':
    app.run(host='10.12.1.206', debug=True, port=5000)