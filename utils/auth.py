import utils.io as io
import sodapy


def get_soda_client(url):
    return sodapy.Socrata(url, None)



def get_private_key(key_type='googlemaps_key',cfg=io.load_config_file()):
    key_json = io.load_file(name=key_type,cfg=cfg)
    return key_json['private_key']


def get_client_id(key_type='googlemaps_key',cfg=io.load_config_file()):
    key_json = io.load_file(name=key_type,cfg=cfg)
    return key_json['client_id']


if __name__== "__main__":
    pass
