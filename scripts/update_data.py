from rootpath import *
from utils import data


def main():
    _ = data.load_bike_data(update=True, save_data=True,verbose=True)


if __name__=="__main__":
    main()