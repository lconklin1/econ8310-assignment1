from pygam import LinearGAM, s
import pandas as pd
import patsy as pt
import numpy as np

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['day_of_week_num'] = pd.to_datetime(data['Timestamp']).dt.weekday+1
data['hour_modified'] = data['hour']+1

data_test = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")
data_test['day_of_week_num'] = pd.to_datetime(data_test['Timestamp']).dt.weekday+1
data_test['hour_modified'] = data_test['hour']+1
data_test['trips'] = 0

eqn = """trips ~ -1 + month + day_of_week_num + hour"""
y,x = pt.dmatrices(eqn, data=data)
gam = LinearGAM(s(0) + s(1) + s(2) )
gam = gam.gridsearch(np.asarray(x), y)
model = gam
modelFit = model.fit(data[['month','day_of_week_num','hour']], data['trips'])
data_test['trips'] = data_test.apply(lambda row: modelFit.predict([[row.month, row.day_of_week_num, row.hour_modified]])[0], axis=1)

pred = data_test['trips'].values 
