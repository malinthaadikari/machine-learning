import pandas as pand
import os.path

resource_path = os.path.join(os.path.split(__file__)[0], "resources/datasets/")


def get_dataframe(filename):
    df = pand.read_csv(resource_path+filename)
    return df