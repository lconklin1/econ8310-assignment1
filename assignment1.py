from pygam import LinearGAM, s#, f, l
import pandas as pd
import patsy as pt
import numpy as np
#from plotly import subplots
#import plotly.offline as py
#import plotly.graph_objs as go
#import plotly.express as px

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data_test = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")

data_combined = pd.concat([data,data_test],ignore_index=False)
data_combined.set_index(pd.DatetimeIndex(data_combined['Timestamp']), inplace=True)
data_combined.fillna(0,inplace=True)

data_combined['day_of_week'] = pd.to_datetime(data_combined['Timestamp']).dt.day_name()
data_combined['day_of_week_num'] = pd.to_datetime(data_combined['Timestamp']).dt.weekday+1
data_combined['hour_modified'] = data_combined['hour']+1

data['day_of_week'] = pd.to_datetime(data['Timestamp']).dt.day_name()
data['day_of_week_num'] = pd.to_datetime(data['Timestamp']).dt.weekday+1
data['hour_modified'] = data['hour']+1

# Generate x and y matrices
eqn = """trips ~ -1 + month + day_of_week_num + hour"""
y,x = pt.dmatrices(eqn, data=data)

# Initialize and fit the model
gam = LinearGAM(s(0) + s(1) + s(2) )
gam = gam.gridsearch(np.asarray(x), y)

data_combined['trips_forecasted'] = data_combined.apply(lambda row: gam.predict([[row.month, row.day_of_week_num, row.hour_modified]])[0], axis=1)
