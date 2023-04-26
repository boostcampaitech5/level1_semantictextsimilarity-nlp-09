import yaml
from omegaconf import OmegaConf
from constants import CONFIG

class Config():
    def __init__(self, cfg, opt, model_dir):
        self.batch_size = int(cfg[opt]["batch_size"])
        self.max_epoch = int(cfg[opt]["max_epoch"])
        self.shuffle = bool(cfg[opt]["shuffle"])
        self.learning_rate = float(cfg[opt]["learning_rate"])
        self.sweep = bool(cfg[opt]["sweep"])
        self.num_workers = int(cfg[opt]["num_workers"])
        self.hidden_dropout_prob = float(cfg[opt]["hidden_dropout_prob"])
        self.attention_probs_dropout_prob = float(
            cfg[opt]["attention_probs_dropout_prob"])
        
        self.model_name = model_dir["model"]["name"]
        # self.model_name = str(cfg[opt]["model"])

        self.train_path = model_dir["model"]["train_path"]
        self.dev_path = model_dir["model"]["dev_path"]
        self.test_path = model_dir["model"]["test_path"]
        self.predict_path = model_dir["model"]["predict_path"]
        
    def set_folder_dir(self, folder_dir):
        self.folder_dir = folder_dir

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return Config(config, opt, model_dir)


def load_sweep_config(config, opt, model):

    with open(model) as file:
        model_dir = yaml.safe_load(file)

    return Config(config, opt, model_dir)

# import omegaconf
def load_omegaconf():
    return OmegaConf.load(CONFIG.CONFIG_PATH)
