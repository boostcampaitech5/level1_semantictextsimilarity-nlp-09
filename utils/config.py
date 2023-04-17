import yaml


class Config():
    def __init__(self, cfg, opt):
        self.batch_size = int(cfg[opt]["batch_size"])
        self.max_epoch = int(cfg[opt]["max_epoch"])
        self.shuffle = bool(cfg[opt]["shuffle"])
        self.learning_rate = float(cfg[opt]["learning_rate"])
        self.sweep = bool(cfg[opt]["sweep"])
        self.loss = cfg[opt]["loss"]

        self.model_name = cfg[opt]["name"]
        self.train_path = cfg[opt]["train_path"]
        self.dev_path = cfg[opt]["dev_path"]
        self.test_path = cfg[opt]["test_path"]
        self.predict_path = cfg[opt]["predict_path"]

    def set_folder_dir(self, folder_dir):
        self.folder_dir = folder_dir


def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return Config(config, "train"), Config(config, "inference")
