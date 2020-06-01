from flask import Flask, request, redirect, url_for, flash, jsonify
# AutoML library
import logging
import sys

import matplotlib.pyplot as plt
import json
import pandas as pd
import gcsfs

import google.cloud.automl_v1beta1.proto.data_types_pb2 as data_types
#import setpath
from google.cloud import automl_v1beta1 as automl

#added for blueprint
from flask import Blueprint

# from google.cloud import automl_v1beta1 as automl
from google.cloud.automl_v1beta1 import enums

#added for blueprint
googleautoml_api = Blueprint('googleautoml_api', __name__)

class googleautoml:
    
    COMPUTE_REGION = "us-central1"
      
    client = automl.AutoMlClient.from_service_account_file('jovial-talon-263619-b9acd4414024.json')
    prediction_client = automl.PredictionServiceClient.from_service_account_file('jovial-talon-263619-b9acd4414024.json')      
    dataset_name = ""
            
    # Get the GCP location of your project.
    def getgcplocation(project_id):       
        return googleautoml.client.location_path(project_id, googleautoml.COMPUTE_REGION)  
        

    # List datasets in Project
    # Note: In a dropdown all the available list of datasets should be displayed.
    # We can either save the list of datasets in db or fetch it from the project location    
    @googleautoml_api.route('/api/list_dataset', methods=['POST'])
    def getListOfDataset(): 

        project_id = request.json['projectid']        

        project_location = googleautoml.getgcplocation(project_id)
        
        logging.info("The value of project_location: {}".format(project_location))        
        list_datasets = googleautoml.client.list_datasets(project_location)        
        datasets = { dataset.display_name: dataset.name for dataset in list_datasets }                
        list_models = googleautoml.client.list_models(project_location)        
        models = { model.display_name: model.name for model in list_models }        
        logging.info("List of models that are present in the project location: {}".format(models))   
        
        return 'ok'  
        
    

    # create dataset  - used in import dataset method  
    def createDataset(dataset_display_name, project_location):        
        dataset_settings = {'display_name': dataset_display_name, 
                            'tables_dataset_metadata': {}}
        create_dataset_response = googleautoml.client.create_dataset(project_location, dataset_settings)        
        logging.info("Value of create_dataset_response: {}".format(create_dataset_response))        
        return create_dataset_response.name
        
    # used in import dataset method
    def inputConfig(bucket_name, train_input_uris):                  
        gcs_input_uris = []
        gcs_input_uris.append('gs://' + bucket_name + '/' + train_input_uris )        
        logging.info("URI for the training input data: {}".format(gcs_input_uris))

        # Define input configuration.
        input_config = {
            'gcs_source': {
                'input_uris': gcs_input_uris
            }
        }        
        logging.info("Value of Input config: {}".format(input_config))
        return input_config

    # import data into the dataset
    def importDataset(bucket_name, train_input_uris, dataset_display_name):  
        
        # project_id = request.json['projectid']
        project_id = "jovial-talon-263619"        

        project_location = googleautoml.getgcplocation(project_id) 
        
        # get dataset name
        dataset_name = googleautoml.createDataset(dataset_display_name, project_location)
        
        # get input config    
        input_config = googleautoml.inputConfig(bucket_name, train_input_uris)

        # importing data into the dataset
        import_data_response = googleautoml.client.import_data(dataset_name, input_config)
        # Synchronous check of operation status. Wait until import is done.
        import_data_result = import_data_response.result()
        import_data_response.done()

        #get the Dataset details
        dataset = googleautoml.client.get_dataset(dataset_name) 
        

    def listColumnTableSpecs(dataset_name):        
        # List table specs
        list_table_specs_response = googleautoml.client.list_table_specs(dataset_name)
        table_specs = [s for s in list_table_specs_response]        

        # List column specs
        table_spec_name = table_specs[0].name
        list_column_specs_response = googleautoml.client.list_column_specs(table_spec_name)
        column_specs = {s.display_name: s for s in list_column_specs_response}        
        return column_specs
        

    def getFeature():
        # Print Features and data_type
        features = [(key, data_types.TypeCode.Name(value.data_type.type_code)) for key, value in column_specs.items()]        
        logging.info("Feature list:")
        for feature in features:
            print(feature[0],':', feature[1])

    # draw pie chart
    def drawPieChart(self):        
        type_counts = {}
        for column_spec in column_specs.values():
            type_name = data_types.TypeCode.Name(column_spec.data_type.type_code)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
        plt.pie(x=type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        plt.axis('equal')
        plt.show()

    
    # select a target column  
    @googleautoml_api.route('/api/assign_label', methods=['POST'])  
    def assignLabel():        
        
        column_name = request.json['target_column_name'] 
        model_display_name = request.json['model_display_name']        
        displayName = request.json['dataset_display_name']
        dataset_name = ""
        project_id = "jovial-talon-263619" 

        try:
            project_location = googleautoml.getgcplocation(project_id)
        
            logging.info("The value of project_location: {}".format(project_location))        
            list_datasets = googleautoml.client.list_datasets(project_location)                

            for dataset in list_datasets:
                if dataset.display_name == displayName:
                    dataset_name = dataset.name
            

            column_specs = googleautoml.listColumnTableSpecs(dataset_name)     


            
            if (googleautoml.list_models(project_id,googleautoml.COMPUTE_REGION, model_display_name)): 
                # print("Model already exist")      
                return "Model already exist"                        
                
            else:
                #print("Model needs to be created")
                model_name = googleautoml.createModel(model_display_name, project_id, dataset_name, column_specs, column_name, displayName) 
                #print("Model created.Model Name: ",model_name)  
                return 'Model created successfully @ {}'.format(model_name.name)              
                #return model_name   
        except Exception as e: 
            if "https" in str(e):
                error_statement = str(e).partition(": ")[2]            
            else:
                error_statement = str(e)
            logging.error(str(e))            
            return error_statement                  
            

    
    def createModel(model_display_name, project_id, dataset_name, column_specs, column_name, displayName):        
        try:
            project_location = googleautoml.getgcplocation(project_id)       
        
            print("The value of column Name: ",column_name)
            
            # train_budget_milli_node_hours : specifies the train budget of creating this model i.e. values in this field
            # means 1 node hour. The training cost of the model will not exceed this budget.
            # The train budget must be between 1,000 and 72,000 milli node hours, inclusive.
            model_dict = {
                'display_name': model_display_name,
                'dataset_id': dataset_name.rsplit('/', 1)[-1],
                'tables_model_metadata': {
                    'train_budget_milli_node_hours': 1000,
                    #'target_column_spec': column_specs['ActionTaken']
                    'target_column_spec': column_specs[column_name]
                    }
            }        
            #logging.info("Model dict: {}".format(model_dict))                
            create_model_response = googleautoml.client.create_model(project_location, model_dict)
            # Wait until model training is done.
            create_model_result = create_model_response.result()
            
            #evaluate model
            
            # # Get complete detail of the model.
            model_name = create_model_result.name         
            # logging.info("The model name is: {}".format(googleautoml.client.get_model(model_name)))
            model_id = model_name.rsplit('/', 1)[-1]
            #print("********** Model Id: ", model_id)
            #googleautoml.display_evaluation(model_id)        
            #googleautoml.modelDeployment(model_name)    
            return googleautoml.client.get_model(model_name)
        
        except Exception as e: 
            if "https" in str(e):
                error_statement = str(e).partition(": ")[2]            
            else:
                error_statement = str(e)
            logging.error(str(e))            
            return error_statement
         


    #@googleautoml_api.route('/api/display_eval', methods=['POST'])      
    def display_evaluation(model_id):
        
        from google.cloud import automl_v1beta1 as automl

        client = automl.AutoMlClient()
        
        project_id = 'jovial-talon-263619'
        compute_region = 'us-central1'
        # model_display_name = request.json['model_name']
        #model_id = request.json['model_id']

        # Get the full path of the model.
        model_full_id = client.model_path(project_id, compute_region, model_id)
        
             

        # List all the model evaluations in the model by applying filter.
        #response = client.list_model_evaluations(model_full_id, filter_)
        response = client.list_model_evaluations(model_full_id)#, filter_)

        # Iterate through the results.
        for element in response:
            # There is evaluation for each class in a model and for overall model.
            # Get only the evaluation of overall model.                        
            if not element.annotation_spec_id:
                model_evaluation_id = element.name.split("/")[-1]

        # Resource name for the model evaluation.
        model_evaluation_full_id = client.model_evaluation_path(
            project_id, compute_region, model_id, model_evaluation_id
        )
        #print("****************** Model Evaluation Full Id: ", model_evaluation_full_id)

        
        # Get a model evaluation.
        model_evaluation = client.get_model_evaluation(model_evaluation_full_id)
        
        classification_metrics = model_evaluation.classification_evaluation_metrics  
        if str(classification_metrics):
            confidence_metrics = classification_metrics.confidence_metrics_entry

            # Showing model score based on threshold of 0.5
            print("Model classification metrics (threshold at 0.5):")
            for confidence_metrics_entry in confidence_metrics:
                if confidence_metrics_entry.confidence_threshold == 0.5:
                    print(
                        "Model Precision: {}%".format(
                            round(confidence_metrics_entry.precision * 100, 2)
                        )
                    )
                    print(
                        "Model Recall: {}%".format(
                            round(confidence_metrics_entry.recall * 100, 2)
                        )
                    )
                    print(
                        "Model F1 score: {}%".format(
                            round(confidence_metrics_entry.f1_score * 100, 2)
                        )
                    )
            print("Model AUPRC: {}".format(classification_metrics.au_prc))
            print("Model AUROC: {}".format(classification_metrics.au_roc))
            print("Model log loss: {}".format(classification_metrics.log_loss))

        regression_metrics = model_evaluation.regression_evaluation_metrics
        if str(regression_metrics):
            print("Model regression metrics:")
            print("Model RMSE: {}".format(
            regression_metrics.root_mean_squared_error
            ))
            print("Model MAE: {}".format(regression_metrics.mean_absolute_error))
            print("Model MAPE: {}".format(
                regression_metrics.mean_absolute_percentage_error))
            print("Model R^2: {}".format(regression_metrics.r_squared))

        
        
        #return "Ok"

        # [END automl_tables_display_evaluation]
    


    
    @googleautoml_api.route('/api/model_deployment', methods=['POST'])
    def modelDeployment():
        try:
            from google.cloud import automl
            client = automl.AutoMlClient()
            uri = request.json['uri']
            strings = uri.split('/')
            model_name = strings[5]
            project_id = "jovial-talon-263619"
            google_location = strings[3]
            model_full_id = client.model_path(project_id, google_location, model_name)
            #deploy_model_response = googleautoml.client.deploy_model(model_name)
            response = client.deploy_model(model_full_id)
            #deploy_model_result = deploy_model_response.result()
            #googleautoml.client.get_model(model_name)
            print("Model deployment finished. {}".format(response.result()))
            return "Model deployed successfully"
        except Exception as e: 
            if "https" in str(e):
                error_statement = str(e).partition(": ")[2]            
            else:
                error_statement = str(e)
            logging.error(str(e))            
            return error_statement

            
    @googleautoml_api.route('/api/model_undeployment', methods=['POST'])
    def modelUnDeployment():
        try:
            from google.cloud import automl
            client = automl.AutoMlClient()
            uri = request.json['uri']
            strings = uri.split('/')
            model_name = strings[5]
            project_id = "jovial-talon-263619"
            google_location = strings[3]
            model_full_id = client.model_path(project_id, google_location, model_name)
            #deploy_model_response = googleautoml.client.deploy_model(model_name)
            response = client.undeploy_model(model_full_id)
            #deploy_model_result = deploy_model_response.result()
            #googleautoml.client.get_model(model_name)
            print("Model Undeployment finished. {}".format(response.result()))
            return "Model Undeployed successfully"
        except Exception as e: 
            if "https" in str(e):
                error_statement = str(e).partition(": ")[2]            
            else:
                error_statement = str(e)
            logging.error(str(e))            
            return error_statement    

    @googleautoml_api.route('/api/online_prediction', methods=['POST'])        
    def online_predict(feature_importance=None):
        # [START automl_tables_predict]
        # TODO(developer): Uncomment and set the following variables
        project_id = "jovial-talon-263619"
        compute_region = request.json['compute_region']
        model_display_name = request.json['model_name']
        #inputs = {'id': 758476,'I10_DX1': 'B349','I10_DX2': 'R509','I10_DX3': 'R42','I10_DX4': 'R000','I10_DX5': 'R0682','CPT1': 36415,'CPT2': 71020,'CPT3': 80053,'CPT4': 81001,'CPT5': 82550,'MOD1': '0.0','MOD2': '0.0','MOD3': '0.0','MOD4': '0.0','MOD5': '0.0'}
        values = request.json['values']
        inputs = eval(values)
        res=[]

        from google.cloud import automl_v1beta1 as automl

        client = automl.TablesClient(project=project_id, region=compute_region)

        if feature_importance:
            response = client.predict(
                model_display_name=model_display_name,
                inputs=inputs,
                feature_importance=True,
            )
        else:
            response = client.predict(
                model_display_name=model_display_name, inputs=inputs
            )

        print("Prediction results:")
        for result in response.payload:
            print(
                "Predicted class name: {}".format(result.tables.value.string_value)
            )
            print("Predicted class score: {}".format(result.tables.score))
            res.append(result.tables.value.string_value)
            res.append(result.tables.score)

            if feature_importance:
                # get features of top importance
                feat_list = [
                    (column.feature_importance, column.column_display_name)
                    for column in result.tables.tables_model_column_info
                ]
                feat_list.sort(reverse=True)
                if len(feat_list) < 10:
                    feat_to_show = len(feat_list)
                else:
                    feat_to_show = 10

                print("Features of top importance:")
                for feat in feat_list[:feat_to_show]:
                    print(feat)
        def Convert(res):
            res_d = {res[i]: res[i+1] for i in range(0, len(res),2)}
            return res_d
        res_dict = Convert(res)
        pred_res = max(res_dict, key=res_dict.get)
        return "Online Prediction Success.\n Predicted label is:\n {}".format(pred_res)
    
    
    #@googleautoml_api.route('/api/model_download', methods=['POST'])
    def model_export():        
        from google.cloud.automl_v1beta1 import __package__


        project_id = 'jovial-talon-263619'
        output_uri = request.json['export_location']
        name = request.json['model_name']

        gcs_output_directory = automl.types.GcsDestination(
                                    output_uri_prefix=output_uri)

        print("GCS o/p directory: ",gcs_output_directory)
        output_config = googleautoml.client.get_model(name).ModelExportOutputConfig(
                gcs_output_directory = "gs://test_bucket_dec3"
        )
        print("Output config: ",output_config)        
        
        response = googleautoml.client.export_model(name, output_config)
        print(response.result())
        
        return "Model exported"

    @googleautoml_api.route('/api/batch_prediction', methods=['POST'])
    #def makeBatchPrediction(bucket_name, predict_gcs_input_uris):
    def makeBatchPrediction():

        bucket_name = request.json['bucket_name']            
        model_name = request.json['model_name']  
        source_file_name = request.json['src_file_name_destination']  
        predict_gcs_input_uris = source_file_name.split("\\")[-1]

        try:
            from main_google import bucket
            bucket.upload_prediction_blob(bucket_name, predict_gcs_input_uris, source_file_name)        
            
            batch_predict_gcs_input_uris = []
            batch_predict_gcs_input_uris.append('gs://' + bucket_name + '/' + predict_gcs_input_uris )
            
            print("The testing datasource from google cloud storage: {}".format(batch_predict_gcs_input_uris))       


            batch_predict_gcs_output_uri_prefix = 'gs://' + bucket_name + '/'        
            print("Output file path for the predicted file: {}".format(batch_predict_gcs_output_uri_prefix))       

            # Define input source.
            batch_prediction_input_source = {
            'gcs_source': {
                'input_uris': batch_predict_gcs_input_uris
            }
            }
            # Define output target.
            batch_prediction_output_target = {
                'gcs_destination': {
                'output_uri_prefix': batch_predict_gcs_output_uri_prefix
                }
            }

            print("Batch prediction output target: ", batch_prediction_output_target)
            # Launch batch prediction.
            batch_predict_response = googleautoml.prediction_client.batch_predict(
                model_name, batch_prediction_input_source, batch_prediction_output_target)        
            print('Batch prediction operation: {}'.format(batch_predict_response.operation))       
            # Wait until batch prediction is done.
            batch_predict_result = batch_predict_response.result()            
                                   
            logging.info('Batch predict response metadata: {}'.format(batch_predict_response.metadata))   
            metadata = '{}'.format(batch_predict_response.metadata)            

            sub = metadata.split('gcs_output_directory: "',1)[1]            
            sub = sub.split('"',1)[0]
            

            dest_test_file = sub + "/tables_1.csv"           
                       

            fs = gcsfs.GCSFileSystem(project='jovial-talon-263619')
            with fs.open(dest_test_file) as f:
                df = pd.read_csv(f)
            
            df2=df[df.columns[-3:]]
            df2.columns = ['None','D011','D097']
            df["Predicted_label"]= df2.idxmax(axis=1)
            rem_char=[':','/','//','.','-']
            for i in rem_char:
                sub=sub.replace(i,'')
            df.to_csv('D:\\PredictionResult\\Google\\{}_prediction_google.csv'.format(sub))
            
            pred_json = df.to_json(orient='records')
            var = 'prediction success@'
            return  '{} {}'.format(var, pred_json)
            #return pred_json
        except Exception as e: 
            if "https" in str(e):
                error_statement = str(e).partition(": ")[2]            
            else:
                error_statement = str(e)
            logging.error(str(e))            
            return error_statement

        
    
    
    
    # shows us the list of models
    def list_models(project_id, compute_region, model_display_name):
        """List all models."""
        
        # # from google.cloud import automl_v1beta1 as automl
        # from google.cloud.automl_v1beta1 import enums
        # global model_already_deployed, 
        model_already_deployed = ""
        client = automl.AutoMlClient()

        # A resource that represents Google Cloud Platform location.
        project_location = client.location_path(project_id, compute_region)

        # List all the models available in the region by applying filter.
        response = client.list_models(project_location)        
        logging.info('List of models:')       
        for model in response:
            if model.deployment_state == enums.Model.DeploymentState.DEPLOYED:
                if model.display_name == model_display_name:
                    model_name = model.name                    
                    logging.info('Model is already deployed: {}'.format(model_name))       
                    model_already_deployed = True  
                    break                               
                else:
                    model_already_deployed = False                    
                

        return model_already_deployed

    
 
def main():
    automlinstnc = googleautoml(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    automlinstnc.getgcplocation()
    automlinstnc.getListOfDataset()
    automlinstnc.createDataset()
    automlinstnc.inputConfig()
    # takes most of the time in importing the dataset
    automlinstnc.importDataset()
    automlinstnc.listColumnTableSpecs()
    automlinstnc.getFeature()
    automlinstnc.drawPieChart()
    automlinstnc.assignLabel()
    
if __name__ == "__main__":
    app.run(host='10.12.1.206', debug=True, port=5000)
    #main()
    # python file_name projectid bucket_name train_input_uri predict_input_uri model_display_name
    # to execute: python newautomlexecutemodified.py dcsautoml buckt_demo1 census_income.csv census_income_batch_prediction_input.csv census_income_model_sep19
