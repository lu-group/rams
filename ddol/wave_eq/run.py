from training import run as training_vanilla
from training_rar_sampling import run as training_rams
from evaluation import run as evaluation
training_vanilla()
training_rams()
evaluation()