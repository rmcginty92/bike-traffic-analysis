from visual_modules import *
from computational_modules import *

import utils.io as io
import utils.alg as alg
import utils.gen as gen
import utils.features as features

desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option('expand_frame_repr',True)


cfg = io.load_config_file()

# get paths
pwd = io.get_path('project_path',cfg=cfg)


# Data

xdf = io.load_file('feature_file')

xdf1 = xdf.groupby(['day','month','year']).mean()
