import configparser

class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))
        
        #Hyper-parameter
        self.model_num = conf.getint("Model_Setup", "model_num")
        self.relation_select = conf.get("Model_Setup", "relation_select")
        self.random_seed = conf.get("Model_Setup", "random_seed")
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.hidden_dimension = conf.getint("Model_Setup", "hidden_dimension")
        self.dropout = conf.getfloat("Model_Setup", "dropout")
        self.alpha = conf.getfloat("Model_Setup", "alpha")
        self.beta = conf.getfloat("Model_Setup", "beta")
        self.gamma = conf.getfloat("Model_Setup", "gamma")
