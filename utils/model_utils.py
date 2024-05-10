from models.lstm_model import LSTM
from models.model_test import CNN

import config


def model_factory(model_type):
    model_classes = {
        'LSTM': LSTM,
        'CNN': CNN
    }
    try:
        model_class = model_classes[model_type]
        return model_class(config.model_config['input_size'], config.model_config['hidden_layers'], config.model_config['output_size'])
    except KeyError:
        raise ValueError(f"Model type {model_type} not supported")

model = model_factory(config.model_config['model_type'])

def save_model():
    pass
def load_model():
    pass