# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:07:45 2019

@author: datacore
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:01:13 2019

@author: datacore
"""

from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_restful import Resource
import os
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

#added for blueprint
from flask import Blueprint

#added for blueprint
h2oautoml_api = Blueprint('h2oautoml_api', __name__)

app = Flask(__name__)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

@h2oautoml_api.route('/api/WS_create_h2o', methods=['POST'])
def WSCreate():
        ip = request.json['ip']
        port  = request.json['port']
        cluster_name  = request.json['cluster_name']
        nthreads  = request.json['nthreads']
        max_mem_size = request.json['max_mem_size']
        
       ##Existing \ new workspace
	   
        ws = h2o.init(ip = ip, name=cluster_name, port = port, nthreads = nthreads, max_mem_size = max_mem_size )
        print(type(nthreads))
		#print(type(nthreads))
        return "Workspace successfully allocated"
        
@h2oautoml_api.route('/api/upload_data_h2o', methods=['POST'])
def UploadCSV():
        file = request.json['file_path']
        print(file)
        data = h2o.import_file(file)
        #des_data = data.describe()
        print(data)
        return "file uploaded successfully" 
    
@h2oautoml_api.route('/api/AutoMLRun_h2o', methods=['POST'])
def RunAutoML():   
        file = request.json['file_path']
        max_models = request.json['max_models']
        max_runtime_secs  = request.json['max_runtime_secs']
        seed  = request.json['seed']
        ip = request.json['ip']
        port  = request.json['port']
        nthreads  = request.json['nthreads']
        max_mem_size = request.json['max_mem_size']
        target_var = request.json['target_var']
        best_model = request.json['best_model']
        cluster_name  = request.json['cluster_name']
       ##Existing \ new workspace
        h2o.init(ip = ip, port = port ,name=cluster_name, nthreads = nthreads, max_mem_size = max_mem_size )
        print('Found existing Workspace.')
        data = h2o.import_file(file)
        predictors = list(data.columns) 
        predictors.remove(target_var)  # Since we need to predict quality
        print(predictors)
        try:
            aml = H2OAutoML(max_models = max_models, max_runtime_secs=max_runtime_secs, seed = seed,exclude_algos = ["XGBoost","DeepLearning"])
            aml.train(x=predictors, y=target_var, training_frame=data)
            #print(aml.leaderboard)
            aml_lb = aml.leaderboard
            print(aml_lb)
            dff = aml_lb.as_data_frame()
            #print(dff)
            # changing index cols with rename() 
            dff.rename(index = {0: "one", 1: "two",2: "three",3: "four",4: "five"}, 
                                     inplace = True) 
            dff_json = dff.to_json(orient='index')
            print(dff_json)
            best_model_id = aml.leader.model_id
            best_model_id
            var1 = "@"
            var2 = var1 + best_model_id
            #best_model = "Model_h2o"
            # save the model
            # Join various path components 
            # Path 
            cwd = 'D:\\DCSAIAUTOML\\BestModels\\h2o'
            #best_model = "Model_h2o"
            model_path = os.path.join(cwd, best_model)
            print(model_path)
            my_model = h2o.save_model(model=aml.leader, path=model_path, force=True)
            modelfile = aml.download_mojo(path=model_path, get_genmodel_jar=True)
            print("Model saved to " + modelfile)
            #return dff_json
            #custom_train_1(file,target_var)
            h2o.shutdown(prompt=False)
            return  '{} {}'.format(dff_json, var2)
        
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            return error_statement
    
        
@h2oautoml_api.route('/api/prediction_h2o', methods=['POST'])
def PredictionH2o():
        best_model = request.json['best_model']
        best_model_id = request.json['best_model_id']
        file = request.json['test_file_path']
        target_var = request.json['target_var']
        ip = request.json['ip']
        port  = request.json['port']
        nthreads  = request.json['nthreads']
        max_mem_size = request.json['max_mem_size']
        cluster_name  = request.json['cluster_name']
        
       ##Existing \ new workspace
	   
        h2o.init(ip = ip, port = port ,name=cluster_name, nthreads = nthreads, max_mem_size = max_mem_size )
        # load the model
        import os
        try:
            cwd = 'D:\\DCSAIAUTOML\\BestModels\\h2o'
            model_path = os.path.join(cwd, best_model, best_model_id)
            print(model_path)
            saved_model = h2o.load_model(model_path)
            #des_data = data.describe()

            print(file)
            test_data = h2o.import_file(file)
            #des_data = data.describe()
            #print(test_data)
            #predictors_test = list(test_data.columns) 
            #predictors_test.remove(target_var)
            #predictors_df = test_data.as_data_frame()

            preds = saved_model.predict(test_data)
            #print(preds)
            dff = preds.as_data_frame()
            #dff1 = dff.drop(['p51'], axis=1)
            #dff2 = dff1.rename(columns={"predict": "Prediction", "Churner": "Churner Probability", "Nonchurner": "Nonchurner Probability"}) 
            test_df = test_data.as_data_frame()
            #stock_df = test_df[['fund_id','stock_id','ActionTaken']]
            result = pd.concat([test_df,dff],axis=1)
            #result.head()
            result.to_csv('D:/PredictionResult/H20/Prediction_h2o.csv', index=False, date_format='%Y%m%d')

            #pred_df.rename(index = {"predict": "prediction", "p0": "Prob for class1","p1": "Prob for class2",
                                #"p2": "Prob for class3"}, inplace = True) 
            pred_json = result.to_json(orient='records')
            #print(pred_json)
            h2o.shutdown()
            return pred_json
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            return error_statement

if __name__ == '__main__':
    app.run(host='10.12.1.206', debug=True, port=54321)