# Configuration settings for the CNN model

# Model configuration
MODEL_TYPE = 'CNN'
INPUT_SHAPE = (64, 64, 3)  # Example input shape for images
NUM_CLASSES = 2  # Number of output classes (binary classification)

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Flask application configuration
FLASK_ENV = 'development'
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000
