import os
import inspect
import sys

# add the 'current' directory as one where we can import modules
src_dir = os.path.split(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))[0]
sys.path.append(src_dir)
