import os

# Пути
DATA_PATH = os.path.join("data", "rus.txt")
MODEL_PATH = "best_model.pth"
LOG_DIR = "runs"

# Гиперпараметры модели
EMBED_DIM = 512
NUM_HEADS = 8
FF_DIM = 2048
NUM_LAYERS = 6
DROPOUT = 0.1

# Обучение
BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE = 5
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.98)
EPS = 1e-9

# Токенизация
VOCAB_SIZE = 30000
MAX_LEN = 256

# Seed для воспроизводимости
SEED = 42
