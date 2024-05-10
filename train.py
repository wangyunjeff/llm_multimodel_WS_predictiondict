# train.py
from utils.model_utils import model_factory
import config

def main():
    model_type = 'ModelA'  # This could be dynamically set or parsed from command-line arguments
    model = model_factory(model_type)

    # Access training parameters
    learning_rate = config.training_params['learning_rate']
    batch_size = config.training_params['batch_size']
    num_epochs = config.training_params['num_epochs']

    # Training logic here

if __name__ == "__main__":
    main()
