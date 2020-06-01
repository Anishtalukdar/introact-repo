import os
import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model
#import h2o
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

#h2o.init(ip="localhost", port=54321)

input_sample = pd.DataFrame({'I10_DX1': pd.Series(['R45851'], dtype='object'), 'I10_DX2': pd.Series(['M25571'], dtype='object'), 'I10_DX3': pd.Series(['F909'], dtype='object'), 'I10_DX4': pd.Series(['F419'], dtype='object'), 'I10_DX5': pd.Series(['Z8669'], dtype='object'), 'CPT1': pd.Series(['99285'], dtype='object'), 'CPT2': pd.Series(['80301'], dtype='object'), 'CPT3': pd.Series(['36832'], dtype='object'), 'CPT4': pd.Series(['80164'], dtype='object'), 'CPT5': pd.Series(['80320'], dtype='object'), 'MOD1': pd.Series(['0.0'], dtype='float64'), 'MOD2': pd.Series(['0.0'], dtype='float64'), 'MOD3': pd.Series(['0.0'], dtype='float64'), 'MOD4': pd.Series(['0.0'], dtype='float64'), 'MOD5': pd.Series(['0.0'], dtype='float64')})
output_sample = np.array([0])

def init():
    global model
    #model_path = Model.get_model_path(model_name = 'bigram_model.pkl')
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model.pkl')
    #model = pickle.load(open(model_path, 'rb'))
    model = joblib.load(model_path)
init()

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})