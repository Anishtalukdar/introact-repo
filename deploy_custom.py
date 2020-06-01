import flask
from flask import request
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/deploy/deploy_custom', methods=['POST'])
def deploy_custom():
        subscription_id = request.json['subscription_id']
        resource_group = request.json['resource_group']
        workspace_name = request.json['workspace_name']
        location = request.json['location']
        deploy_mode = request.json['deploy_mode']  # AKS or ACI        
        exp_name = request.json['exp_name']        #Model file Path        
        reg_model_name = request.json['reg_model_name']        #Model name   
        description = request.json['description']
        score_path = request.json['score_path']   #Python scoring file       
        conda_en_path = request.json['conda_en_path']   #yml file
        service_name = request.json['service_name']
        
        if(deploy_mode=="AKS"):
               cluster_name = request.json['cluster_name']
               
        flg=1
        if(flg==1):        

            import logging
            from matplotlib import pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn import datasets
            from azureml.core import Workspace
            
            import azureml.core
            from azureml.core.experiment import Experiment
            from azureml.core.workspace import Workspace
            from azureml.train.automl import AutoMLConfig
            from azureml.core.compute import AmlCompute
            from azureml.core.compute import ComputeTarget
            from azureml.core.runconfig import RunConfiguration
            from azureml.core.conda_dependencies import CondaDependencies

             
            #ws = Workspace.from_config()
            #!pip install azure.cli
            from azureml.core.authentication import AzureCliAuthentication
            import azure.cli.core
            #cli_auth = AzureCliAuthentication()
            
            ws = Workspace(subscription_id=subscription_id,
                                              resource_group=resource_group,
                                              workspace_name=workspace_name)
                                                        
            print("Found workspace {} at location {}".format(ws.name, ws.location))
            print('Found existing Workspace.')
                        
            cwd = 'D:\\DCSAIAUTOML\\BestModels\\Custom'
            model_path = cwd + '\\' + exp_name + '\\best_model.pkl'
            #model_path = os.path.join(cwd, exp_name)
 
            
            from azureml.core.model import Model
            # Tip: When model_path is set to a directory, you can use the child_paths parameter to include
            #      only some of the files from the directory
            model = Model.register(model_path = model_path,
                                   model_name = reg_model_name,
                                   description = description,
                                   workspace = ws)
                      
            model = Model(ws, name=reg_model_name)
            print(model.name, model.id, model.version, sep = '\t')
                                      
            
            from azureml.core.model import InferenceConfig
            from azureml.core.webservice import AciWebservice
            from azureml.core.webservice import Webservice
            from azureml.core.model import Model
            from azureml.core.environment import Environment
            script_file_name = score_path
            conda_env_file_name = conda_en_path
            
            myenv = Environment.from_conda_specification(name="myenv", file_path=conda_env_file_name)
            inference_config = InferenceConfig(entry_script=script_file_name, environment=myenv)
            print(inference_config)            
            service_name = service_name    
            if(deploy_mode=="ACI"):                        
                    # ACI deploment with-out DOCKER 
                    from azureml.core.model import Model
                    model = Model(ws, id = model.id)
                    
                    try:
                        deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 2)
                        
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
            
            else: #AKS deployment

                    from azureml.core.model import InferenceConfig
                    from azureml.core.webservice import AciWebservice
                    from azureml.core.model import Model
                    from azureml.core.environment import Environment
                    from azureml.core.webservice import AksWebservice, Webservice
               
                    model = Model(ws, id = model.id)
                    
                    from azureml.core.compute import AksCompute, ComputeTarget
                    service_name = service_name
                    aks_target = AksCompute(ws,cluster_name)
                    # If deploying to a cluster configured for dev/test, ensure that it was created with enough
                    # cores and memory to handle this deployment configuration. Note that memory is also used by
                    # things such as dependencies and AML components.
                    try:
                        deployment_config = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 2,enable_app_insights=True,collect_model_data=True,)
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
                               
app.run(host='10.12.1.206', debug=True, port=7726)        