from recovar_torch import RepresentationLearningMultipleAutoencoder
from seisbench_kfold_trainer import SeisBenchKfoldTrainer

MODEL_CLASSES = [RepresentationLearningMultipleAutoencoder]

DATASETS = ["stead"]

NUM_EPOCHS = 20
for train_dataset in DATASETS:
    for model_class in MODEL_CLASSES:
        for split in range(1):
            exp_name = f"recovar_{train_dataset}"
            kfold_trainer = SeisBenchKfoldTrainer(exp_name, model_class, train_dataset, split, epochs=NUM_EPOCHS, apply_resampling=False)
            kfold_trainer.train()
