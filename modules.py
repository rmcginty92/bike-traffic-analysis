import os, sys, json
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import forecastio
from datetime import datetime

from utils import io,gen,alg,data

pwd = os.getcwd()
cfg = io.load_config_file()
date_format, socrata_date_format = io.get_params('date_format','database_date_format',cfg=cfg)

plt.style.use('ggplot')
# Utility Functions
desired_width = 320
pd.set_option('display.width', desired_width)

if __name__ == '__main__':
    pass
