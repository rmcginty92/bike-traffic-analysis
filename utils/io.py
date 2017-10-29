import collections
import inspect
import json
import os
import sys
import numpy as np
import pandas as pd


def load_config_path(path='config', project_name=''):
    # pwd = os.getcwd()
    pwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    dirFound = os.path.isdir(os.path.join(pwd, path))
    while not dirFound and os.path.basename(pwd) != project_name and pwd != os.path.realpath(os.path.join(pwd, '..')):
        pwd = os.path.realpath(os.path.join(pwd, os.pardir))
        dirFound = os.path.isdir(os.path.join(pwd, path))
    # One last check for Directory
    if not os.path.isdir(os.path.join(pwd, path)):
        raise IOError("Could not find {0} directory.".format(path))
    return os.path.join(pwd, path)


def load_config_file(path='config', default_filename='default.cfg', project_name=''):
    # Traverse upwards to find config path
    default_filename = 'default.cfg'
    custom_filename = 'custom.cfg'
    config_path = load_config_path(path=path, project_name=project_name)
    # Get default.cfg
    with open(os.path.join(config_path, default_filename), 'r') as f:
        default = json.load(f)

    if os.path.isfile(os.path.join(config_path, custom_filename)):
        with open(os.path.join(config_path, custom_filename), 'r') as f:
            custom = json.load(f)
        default = update_parameters(default, custom, add_new=True)
    return default

def get_filepath(name, cfg=None):
    if cfg is None: cfg = load_config_file()
    name = cfg["files"].get(name,name)
    return get_path(name)


def get_path(name, forcepath=True, cfg=None):
    if cfg is None: cfg = load_config_file()
    try:
        if not isinstance(name, list):
            path = cfg['paths'].get(name, name)
        else:
            path = name
        if isinstance(path, (str, unicode)) and os.path.splitext(path)[-1] == '.path':  # *.path file
            config_path = load_config_path()
            with open(os.path.join(config_path, path), 'r') as f:
                path = f.read()
            return path
        elif isinstance(path, list):  #
            path_list = list(path)
            path = ''
            for p in path_list:
                if os.path.isdir(os.path.join(path, p)) or (forcepath and p not in cfg['paths']):
                    path = os.path.join(path, p)
                else:
                    path = os.path.join(path, get_path(p, cfg=cfg))
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(get_path(cfg['paths']['project_path'], cfg=cfg), path)
    except:
        raise IOError("Could not resolve path.")


def get_params(*args, **kwargs):
    def _finditem(obj, key):
        if key in obj: return obj[key]
        for k, v in obj.items():
            if isinstance(v, dict):
                item = _finditem(v, key)
                if item is not None:
                    return item

    cfg = kwargs.get('cfg', load_config_file())
    if len(args) > 1:
        res = []
        for arg in args:
            res.append(_finditem(cfg, arg))
    elif len(args) == 1:
        res = _finditem(cfg, args[0])
    else:
        res = None
    return res


def load_key(key_name,cfg=None):
    if cfg is None: cfg = load_config_file()
    res = cfg['keys'].get(key_name, key_name)
    if isinstance(res,list):
        dir_name,filename = res[:-1],res[-1]
        full_filename = os.path.join(get_path(name=dir_name,cfg=cfg),filename)
    else:
        full_filename = os.path.join(get_path(name=get_path('key_path'), cfg=cfg), res)
    if not os.path.isfile(full_filename):
        raise KeyError("{0} not contained in config.json \"keys\".".format(key_name))

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


def load_file(name, asSeries=True, aslist=False, load_param_name=None, cfg=None, **kwargs):
    if cfg is None: cfg = load_config_file()
    res = cfg['files'].get(name, name)
    full_filename = get_path(res)
    ext = os.path.splitext(full_filename)[-1]
    if ext in cfg['extension_mappings']['.csv']:
        if load_param_name is None: load_param_name = name
        csv2df_kwargs = cfg.get('load_kwargs',{}).get(load_param_name, {})
        csv2df_kwargs.update(kwargs)
        res = pd.read_csv(full_filename, **csv2df_kwargs)
        if asSeries and res.ndim > 1 and len(res.columns) == 1: res = res[res.columns[0]]
        if aslist: res = res.values.ravel().tolist()
    elif ext in cfg['extension_mappings']['.json']:
        with open(full_filename, 'r') as f:
            res = json.load(f)
    else:
        with open(full_filename, 'r') as f:
            res = f
    return res


def save_file(name, data, cfg=None, astype=None, **kwargs):
    if cfg is None: cfg = load_config_file()
    if not isinstance(name, list):
        res = cfg['files'].get(name, None)
    else:
        res = name
        name = ''
    if res is None: return False
    if isinstance(res, list):
        dir_name, filename = res[:-1], res[-1]
        dir_name = dir_name[0] if len(dir_name) == 1 else dir_name
    else:
        dir_name, filename = get_path(name='project_path', cfg=cfg), res
    full_filename = os.path.join(get_path(name=dir_name, cfg=cfg), filename)
    ext = os.path.splitext(full_filename)[-1]
    if ext in cfg['extension_mappings']['.csv'] or \
            (astype is not None and astype in cfg['extension_mappings']['.csv']):
        if isinstance(data, (pd.core.frame.DataFrame, pd.core.series.Series)):
            df2csv_kwargs = cfg['save_kwargs'].get(name, {})
            df2csv_kwargs.update(kwargs)
            data.to_csv(full_filename, **df2csv_kwargs)
            return True
    elif ext in cfg['extension_mappings']['.json'] or \
            (astype is not None and astype in cfg['extension_mappings']['.json']):
        with open(full_filename, 'w') as f:
            json.dump(data, f, indent=2)
            return True
    else:
        return False


# --------------------------------------#
###      Config File Operations      ###
# --------------------------------------#


def update_parameters(parameters, new_parameters, add_new=False):
    for k, v in new_parameters.iteritems():
        if isinstance(v, collections.Mapping) and (add_new or k in parameters):
            parameter_subset = parameters.get(k, {})
            r = update_parameters(parameter_subset, v)
            parameters[k] = r
        else:
            parameters[k] = new_parameters[k]
    return parameters


def update_parameter_values(parameters, **kwargs):
    for k, v in parameters.items():
        if isinstance(v, collections.MutableMapping) and k not in kwargs:
            parameters[k] = update_parameter_values(v)
        else:
            if k in kwargs:
                parameters[k] = kwargs[k]
    return parameters


def search_parameters(parameters, *args):
    def _finditem(obj, key):
        if key in obj: return obj[key]
        for k, v in obj.items():
            if isinstance(v, dict):
                item = _finditem(v, key)
                if item is not None:
                    return item

    if len(args) > 1:
        res = []
        for arg in args:
            res.append(_finditem(parameters, arg))
    elif len(args) == 1:
        res = _finditem(parameters, args[0])
    else:
        res = None
    return res


def load_keyword_parameters(parameter_type, include_general_params=False, cfg=None):
    '''

    :param parameter_type: 'general_parameters','waveform_extraction',
                           'trot_run','feature_extraction', or 'classification'
    :param include_general_params: 
    :param cfg: 
    :return: 
    '''
    if cfg is None: cfg = load_config_file()
    parameters = {}
    default_parameters_filename = 'default.parameters'
    custom_parameters_filename = 'custom.parameters'
    parameter_path = get_path('parameters_path', cfg=cfg)

    # full_parameters = load_file(name='default_parameters', cfg=cfg)
    with open(os.path.join(parameter_path, default_parameters_filename), 'r') as f:
        full_parameters = json.load(f)

    # Adding General Parameters
    if include_general_params:
        parameters.update(full_parameters['general_parameters'])
    # Getting default Parameters
    parameters.update(full_parameters[parameter_type])

    # Get custom parameters.
    if os.path.isfile(os.path.join(parameter_path, custom_parameters_filename)):
        custom_param_file = os.path.join(parameter_path, custom_parameters_filename)
        # custom_params = load_file(custom_param_file)
        with open(custom_param_file, 'r') as f:
            custom_params = json.load(f)
        parameters = update_parameters(parameters, custom_params.get(parameter_type, {}))

    # Deciphering
    # parameters = convert_parameter_values(parameters, True)
    return parameters



# --------------------------------------#
###      I/O for feature files       ###
# --------------------------------------#


def interactive_selection(selection_list):
    '''
    User input-based Selection

    :param selection_list: (list, iterable) List of choices
    :return: selected item from list
    '''
    for i, item in enumerate(selection_list):
        print ("({index}): {item}".format(index=i, item=item))
    index = int(raw_input("Select One above: "))
    if 0 < index < len(selection_list):
        selected_item = selection_list[index]
    elif len(selection_list) > 0:
        selected_item = selection_list[0]
    else:
        raise IOError("No files to select.")
    return selected_item


def get_file_list(path, proper_ext=None, n=None):
    '''

    :param path: directory containing content of interest
    :param proper_ext: (str) extension of file type
    :param n:
    :return:
    '''
    try:
        files = np.array(os.listdir(path))
        filenames = np.array([file for file in files if proper_ext is None or proper_ext in os.path.splitext(file)[-1]])
        if len(filenames) == 0:
            return []
        dates = np.array([os.path.getmtime(os.path.join(path, filename)) for filename in filenames])
        if n is None: n = len(dates)
        i_ret = np.argsort(dates)[::-1][:n]
        return filenames[i_ret]
    except:
        return []


def get_directory_list(path, n=None):
    datesanddirs = [(os.path.getmtime(os.path.join(path, dirname)), dirname)
                    for dirname in os.listdir(path)
                    if os.path.isdir(os.path.join(path, dirname))]
    if len(datesanddirs) == 0:
        return []
    if n is None: n = len(datesanddirs)
    datesanddirs.sort(reverse=True)
    return [datesanddir[1] for datesanddir in datesanddirs][:n]


if __name__ == "__main__":
    pass
