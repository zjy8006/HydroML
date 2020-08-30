import json
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from tensorflow import keras
import os
root = os.path.abspath(os.path.dirname('__file__'))
import sys
sys.path.append(root)

from src.SSAUtil import ssa_decomposition


# Your API definition
app = Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def parse_predictors(dataframe,lags_dict,predictor_columns):
    # print('dateframe:{}'.format(dataframe))
    # print('type(lags_dict):{}'.format(type(lags_dict)))
    # print('lags_dict:{}'.format(lags_dict))
    predictors = pd.DataFrame()
    columns = dataframe.columns.values
    # print("columns:{}".format(columns))
    data_size = dataframe.shape[0]
    # print("data_size:{}".format(data_size))
    if type(lags_dict) == int:
        max_lag = lags_dict
    else:
        max_lag = max(lags_dict.values())
    # print("max_lag:{}".format(max_lag))
    samples_size = data_size-max_lag+1
    # print("samples_size:{}".format(samples_size))
    for i in range(len(columns)):
        # Get one input feature
        # print("columns:{}".format(columns[i]))
        one_in = (dataframe[columns[i]]).values  # subsignal
        # print("one_in:{}".format(one_in))
        if type(lags_dict) == int:
            lag=lags_dict
        else:
            lag = lags_dict[columns[i]]
        # print("lag:{}".format(lag))
        oness = pd.DataFrame()  # restor input features
        for j in range(lag):
            # j=0, 0:16-(12-0)+1=0:5
            # j=1, 1:16-(12-1)+1=1:6
            # 
            x = pd.DataFrame(one_in[j:data_size-(lag-j)+1],columns=['X' + str(j + 1)])
            x = x.reset_index(drop=True)
            oness = pd.concat([oness, x], axis=1, sort=False)
        # print("oness=\n{}".format(oness))
        oness = oness.iloc[oness.shape[0]-samples_size:]
        oness = oness.reset_index(drop=True)
        predictors = pd.concat([predictors, oness], axis=1, sort=False)
    predictors = pd.DataFrame(predictors.values,columns=predictor_columns)
    # print("predictors:\n{}".format(predictors))
    last_sample_predictors = predictors.iloc[predictors.shape[0]-1:]
    return last_sample_predictors


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        print("json_:{}".format(json_))
        stationMap = json_["stationMap"]
        target = json_["target"]
        models = json_["models"]
        lead_time = json_["lead_time"]
        historyData = json_["historyData"]
        pred_dates = json_["pred_dates"]

        stations=[]
        for key in historyData.keys():
            stations.append(stationMap[key])

        results = {}
        for stcd in historyData.keys():
            model_results = {}
            station = stationMap[stcd]
            history = historyData[stcd]
            print(history)
            for model in models:
                path = "model/"+station+"/"+target+"/"+model.lower()+"/"+lead_time+"/"
                print("model_path:{}".format(path))
                predictions = {}
                if os.path.exists(path):
                    lags_dict = joblib.load(path+"lags_dict.pkl")
                    # print ('Lags dict loaded')
                    if model.lower().__contains__('lstm'):
                        print("&"*100)
                        lr = keras.models.load_model(path+"model.h5")
                    else:
                        lr = joblib.load(path+"model.pkl") # Load "model.pkl"
                    # print ('Model loaded')
                    predictor_columns = joblib.load(path+"predictor_columns.pkl") # Load "model_columns.pkl"
                    # print(predictor_columns)
                    # print ('Model columns loaded')
                    norm = joblib.load(path+'norm.pkl')
                    # print ('Normalization indicators loaded')
                    if lr:
                        history = list(history)
                        sMin = norm['sMin']
                        sMax = norm['sMax']
                        Y_min = sMin.pop('Y')
                        Y_max = sMax.pop('Y')
                        for pred_date in pred_dates:
                            query = pd.DataFrame(history,columns=["ORIG"])
                            # print("query:\n{}".format(query))
                            if model.__contains__('ssa') or model.__contains__('SSA'):
                                query = ssa_decomposition(query["ORIG"],len(lags_dict))
                                query = query.drop("ORIG",axis=1)
                            # print("query:\n{}".format(query))
                            predictors = parse_predictors(query,lags_dict,predictor_columns)
                            # print("predictors:\n{}".format(predictors))
                            # print(type(query))
                            # print(query)
                            # print('type(predictors):{}'.format(type(predictors)))
                            # print('type(sMin):{}'.format(type(sMin)))
                            # print('type(sMax):{}'.format(type(sMax)))
                            predictors = 2*(predictors-sMin)/(sMax-sMin)-1
                            # print(predictors)
                            if model.lower().__contains__('lstm'):
                                predictors = (predictors.values).reshape(predictors.shape[0], 1, predictors.shape[1])
                            if model.lower().__contains__('lstm') or model.lower().__contains__('DNN'):
                                prediction = lr.predict(predictors).flatten()
                            else:
                                prediction = lr.predict(predictors)
                            # print(type(prediction))
                            prediction = np.multiply(prediction + 1,Y_max - Y_min) / 2 + Y_min
                            if prediction[0]<0.0:
                                prediction[0] = 0.0
                            history.append(prediction[0])
                            predictions[pred_date] = prediction[0]

                    else:
                        # print ('Train the model first')
                        return ('No model here to use')
                        for pred_date in pred_dates:
                            predictions[pred_date]="-"
                else:
                    for pred_date in pred_dates:
                        predictions[pred_date]="-"
                model_results[model] = predictions
            results[stcd] = model_results
            res = json.dumps(results,cls=NumpyEncoder)
            # print(res)
        return jsonify({'predictions': str(res)})
    except:
        return jsonify({'trace': traceback.format_exc()})
        
        



if __name__ == '__main__':
    # try:
    #     port = int(sys.argv[1]) # This is for a command-line input
    # except:
    #     port = 12345 # If you don't provide any port the port will be set to 12345
    app.run(host='0.0.0.0', debug=True)