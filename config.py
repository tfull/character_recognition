import os
import yaml

class Config:
    with open(os.path.dirname(os.path.abspath(__file__)) + "/config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        data_directory = config["data_directory"]
        image_size = config["image_size"]
