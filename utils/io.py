import os, sys, json
import numpy as np
import pandas as pd
from sklearn.externals import joblib


def load_config_path(path='config',project_name=''):
    pwd = os.getcwd()
    dirFound = os.path.isdir(os.path.join(pwd,path))
    while not dirFound and (os.path.basename(pwd)!=project_name or pwd != os.path.realpath(os.path.join(pwd,'..'))):
        pwd = os.path.realpath(os.path.join(pwd,os.pardir))
        dirFound = os.path.isdir(os.path.join(pwd,path))
    # One last check for Directory
    if not os.path.isdir(os.path.join(pwd,path)):
        raise IOError("Could not find {0} directory.".format(path))
    return os.path.join(pwd,path)


def load_config_file(path='config', filename='cfg.json', project_name=''):
    # Traverse upwards to find config path
    config_path = load_config_path(path=path,project_name=project_name)
    with open(os.path.join(config_path,filename),'r') as f:
        return json.load(f)


def get_path(name, forcepath=True, cfg=load_config_file()):
    try:
        if not isinstance(name,list): path = cfg['paths'].get(name,name)
        else: path = name
        if isinstance(path,(str,unicode)) and os.path.splitext(path)[-1]=='.path': # *.path file
            config_path = load_config_path()
            with open(os.path.join(config_path,path),'r') as f:
                path = f.read()
            return path
        elif isinstance(path,list): #
            path_list = list(path)
            path = ''
            for p in path_list:
                if os.path.isdir(os.path.join(path,p)) or (forcepath and not cfg['paths'].has_key(p)):
                    path = os.path.join(path,p)
                else:
                    path = os.path.join(path,get_path(p,cfg=cfg))
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(get_path(cfg['paths']['project_path'],cfg=cfg),path)
    except:
        raise IOError("Could not resolve path.")


def get_filepath(name,cfg=load_config_file()):
    res = cfg['files'][name]
    if isinstance(res,list):
        path_name,file_name = cfg['files'][name]
        if os.path.isabs(path_name): path = path_name
        else: path = get_path(name=path_name,cfg=cfg)
    else:
        if os.path.isabs(res): path,file_name = os.path.split(res)
        else: path, file_name = get_path('project_path',cfg=cfg), res
    return os.path.join(path,file_name)


def get_params(*args,**kwargs):
    def _finditem(obj, key):
        if key in obj: return obj[key]
        for k, v in obj.items():
            if isinstance(v,dict):
                item = _finditem(v, key)
                if item is not None:
                    return item
    cfg = kwargs.get('cfg',load_config_file())
    if len(args) > 1:
        res = []
        for arg in args:
            res.append(_finditem(cfg,arg))
    elif len(args)==1: res = _finditem(cfg,args[0])
    else: res = None
    return res


def load_keyword_parameters(parameter_type, cfg=load_config_file()):
    '''

    :param parameter_type: 'feature_extraction'
    :return:
    '''
    parameter_file = load_file(name='parameters',cfg=cfg)
    return parameter_file[parameter_type]


def load_file(name,cfg=load_config_file(), aslist=False,**kwargs):
    res = cfg['files'].get(name,None)
    if res is None:
        raise KeyError("{0} not contained in config.json \"files\".".format(name))
    if isinstance(res,list):
        dir_name,filename = res[:-1],res[-1]
        full_filename = os.path.join(get_path(name=dir_name,cfg=cfg),filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext in ['.csv','.txt']:
            res = pd.read_csv(full_filename,**kwargs)
            if aslist: res = res.values.ravel().tolist()
        elif ext == '.json':
            with open(full_filename,'r') as f:
                res = json.load(f)
        else:
            with open(full_filename,'r') as f:
                res = f
    else:
        raise TypeError("{0} value is not in correct format.".format(name))
    return res


def load_key(key_name,cfg=load_config_file()):
    res = cfg['keys'].get(key_name, None)
    if res is None:
        raise KeyError("{0} not contained in config.json \"keys\".".format(key_name))
    if isinstance(res,list):
        dir_name,filename = res[:-1],res[-1]
        full_filename = os.path.join(get_path(name=dir_name,cfg=cfg),filename)
    else:
        full_filename = os.path.join(get_path(name=get_path('key_path'), cfg=cfg), res)
    ext = os.path.splitext(full_filename)[-1]
    if ext == '.json':
        with open(full_filename,'r') as f:
            res = json.load(f)
    elif ext == '.key': #
        with open(full_filename,'r') as f:
            res = f.read()
    else:
        raise TypeError("{0} value is not in correct format.".format(key_name))
    return res

def save_file(name,data,cfg=load_config_file(),**kwargs):
    res = cfg['files'].get(name,None)
    if res is None: return False
    if isinstance(res,list):
        dir_name,filename = res
    else:
        filename,dir_name = res,get_path(name='project_path',cfg=cfg)
    full_filename = os.path.join(get_path(name=dir_name,cfg=cfg),filename)
    ext = os.path.splitext(full_filename)[-1]
    if ext in ['.csv','.txt']:
        if isinstance(data,(pd.core.frame.DataFrame,pd.core.series.Series)):
            df2csv_kwargs =kwargs
            df2csv_kwargs.setdefault('index',None)
            data.to_csv(full_filename,**df2csv_kwargs)
            return True
    elif ext == '.json':
        with open(full_filename,'w') as f:
            json.dump(data,f)
            return True
    else:
        return False


def classifier_from_file(clf_filename,ret_build_info=False,ret_test_files=False,verbose=True):
    classifier_path = get_path('classifier_path',forcepath=True)
    test_files_filename = 'valid-test-audio-files.txt'
    clf_name, clf_ext = os.path.splitext(clf_filename)
    clf_path = os.path.join(classifier_path,clf_name)
    if not os.path.isdir(clf_path):
        if verbose: print("ERROR: {0} does not exist.".format(clf_path))
        return None
    # check whether filename is legitimate
    clf_name,clf_ext = os.path.splitext(clf_filename)
    if clf_ext not in ['.pkl','.pickle','.clf']:
        clf_ext = '.pkl'
        clf_filename = ''.join([clf_name,clf_ext])
    if not os.path.isfile(os.path.join(clf_path,clf_filename)):
        if verbose: print("ERROR: {0} does not exist in the following directory:\n{1}".format(clf_filename, clf_path))
        return None
    try:
        clf = joblib.load(os.path.join(clf_path,clf_filename))
        if ret_build_info:
            with open(os.path.join(clf_path, clf_name + '-build_info.json'), 'r') as f:
                build_info = json.load(f)
        else:
            build_info = {}
        if ret_test_files:
            valid_files = pd.read_csv(os.path.join(clf_path,test_files_filename),header=None,index_col=False).values.ravel().tolist()
        else:
            valid_files = []
    except:
        if verbose: print("ERROR: Classifier and build info not loaded properly.")
        return None
    if not (ret_build_info or ret_test_files):
        return clf
    else:
        ret_vars = [clf]
        if ret_build_info: ret_vars.append(build_info)
        if ret_test_files: ret_vars.append(valid_files)
        return ret_vars


def classifier_to_file(clf,clf_filename,Xdf,i_test,build_info,verbose=True):
    classifier_path = get_path('classifier_path')
    sep_char = get_params('sep_char')
    test_files_filename = 'valid-test-audio-files.txt'
    clf_name,clf_ext = os.path.splitext(clf_filename)
    if clf_ext not in ['.pkl','.pickle','.clf']:
        clf_ext = '.pkl'

    # find path
    clf_path = os.path.join(classifier_path,clf_name)
    if not os.path.isdir(clf_path):
        if verbose: print("{0} does not exist, creating directory".format(clf_path))
        if not os.path.isdir(classifier_path):
            os.mkdir(classifier_path)
        os.mkdir(clf_path)
    # getting list of valid test files
    save_full = False
    min_trot_len = 10
    perc_train_test_thresh_range = [0.25,0.75]
    if save_full:
        major_index,_ = zip(*Xdf.index)
        subset_major_index = [fname.split(sep_char)[0] for fname in np.array(major_index)[i_test]]
        test_files = np.unique(np.array(subset_major_index)).tolist()
    else:
        major_index,_ = zip(*Xdf.index)
        sample_counts= pd.Series([fname.split(sep_char)[0] for fname in np.array(major_index)]).value_counts()
        test_sample_counts= pd.Series([fname.split(sep_char)[0] for fname in np.array(major_index)[i_test]]).value_counts()
        sub_sample_counts = sample_counts[test_sample_counts.index]
        perc_train_test = (sub_sample_counts - test_sample_counts).astype(float)/sub_sample_counts
        filtered_sample_counts = test_sample_counts[(test_sample_counts > min_trot_len) &
                                                    (perc_train_test > perc_train_test_thresh_range[0]) &
                                                    (perc_train_test < perc_train_test_thresh_range[1])]
        test_files = filtered_sample_counts.index.tolist()
    try:
        if verbose: print("Saving classifier info...")
        joblib.dump(clf,os.path.join(clf_path,''.join([clf_name,clf_ext])))
        if verbose: print("\t...Classifier Pickled")
        pd.Series(data=test_files).to_csv(os.path.join(clf_path,test_files_filename),index=False,header=None)
        if verbose: print("\t...Test file list saved")
        with open(os.path.join(clf_path,clf_name+'-build_info.json'),'w') as f:
            json.dump(build_info,f)
        if verbose: print("\t...Build info JSON saved.")
    except:
        if verbose: print("ERROR: Classifier and build info not saved properly.")



