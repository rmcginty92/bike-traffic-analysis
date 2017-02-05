import os
import json


def main():
    config_path = os.getcwd()
    cfg_file = os.path.join(config_path,'cfg.json')
    if not os.path.isfile(cfg_file):
        raise IOError('Could not find configuration file: {filename}'.format(filename=cfg_file))
    with open(cfg_file,'r') as f:
        cfg =  json.load(f)
    proj_path_filename = os.path.join(config_path,cfg['paths']['project_path'])
    with open(proj_path_filename,'w') as proj_path_file:
        pardir = os.path.abspath(os.path.join(config_path, os.pardir))
        proj_path_file.write(pardir)
        print("Project path updated.")



    # data_path_filename = os.path.join(config_path,cfg['paths']['data_path'])
    # # Do not update data path file if it already exists.
    # if not os.path.exists(data_path_filename):
    #     with open(os.path.join(config_path,data_path_filename),'w') as data_path_file:
    #         pardir = os.path.abspath(os.path.join(config_path, os.pardir))
    #         data_path_file.write(os.path.join(pardir,'data'))
    #         print("Data path updated.")
    # else:
    #     print("Data path not updated")


if __name__=="__main__":
    main()