# Modules in use
import numpy as np, pandas as pd, scipy as sp
import matplotlib.pyplot as plt
import sodapy, os, json, forecastio
from datetime import datetime

# Utility Functions
import os, sys
plt.style.use('ggplot')
desired_width = 320
pd.set_option('display.width', desired_width)

# Account Info
url = "data.seattle.gov"
dataset_identifier = "65db-xm6k"
client = sodapy.Socrata(url, None)
api_key = '4f6660680e5b0fc85bea5e7fceb25af5'

# System Variables
from pandas.io.tests.parser import index_col
date_format = '%Y-%m-%dT%H:%M:%S'
date_format2 = '%Y-%m-%d %H:%M:%S'
pwd = os.getcwd()
filename = 'bike_data.csv'
lat = 47.649662
lng = -122.350273
json_info_file = reduce(lambda x,y:os.path.join(x,y),[pwd,'Resources','info.json'])
with open(json_info_file,'r') as f:
    info = json.load(f)

if __name__ == '__main__':
    pass
