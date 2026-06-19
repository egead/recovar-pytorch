from recovar_torch import RepresentationLearningMultipleAutoencoder
from kfold_trainer import KfoldTrainer

MODEL_CLASSES = [RepresentationLearningMultipleAutoencoder]

DATASETS = ["instance"]

NUM_EPOCHS = 20
for train_dataset in DATASETS:
    for model_class in MODEL_CLASSES:
        for split in range(1):
            exp_name = f"{train_dataset}_pytorch"
            kfold_trainer = KfoldTrainer(exp_name, model_class, train_dataset, split, epochs=NUM_EPOCHS, apply_resampling=False)
            kfold_trainer.train()
