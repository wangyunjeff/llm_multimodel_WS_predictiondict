# config.py
# Configuration for different models

model_config = {
    'ModelA': {
        'input_size': 784,  # Example for an MNIST dataset
        'hidden_layers': 128,
        'output_size': 10
    },
    'ModelB': {
        'input_size': 150528,  # Example for an ImageNet dataset (224x224x3)
        'hidden_layers': 256,
        'output_size': 1000
    }
}

# You can also include other configurations like training parameters, etc.
training_params = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 10
}
