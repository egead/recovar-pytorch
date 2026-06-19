from recovar_torch import BATCH_SIZE
from kfold_environment import KFoldEnvironment
from directory import *
from os import makedirs
import numpy as np
import pandas as pd
import torch


class KfoldTrainer:
    def __init__(
        self,
        exp_name,
        model_class,
        dataset,
        split,
        epochs,
        apply_resampling=False,
        resampling_eq_ratio=0.5,
        resample_while_keeping_total_waveforms_fixed=False,
        learning_rate=1e-4,
        epsilon=1e-7,
        beta_1=0.99,
        beta_2=0.999,
    ):
        self.exp_name = exp_name
        self.model_class = model_class
        self.dataset = dataset
        self.split = split
        self.epochs = epochs
        self.apply_resampling = apply_resampling
        self.resampling_eq_ratio = resampling_eq_ratio
        self.resample_while_keeping_total_waveforms_fixed = resample_while_keeping_total_waveforms_fixed
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = model_class().name

    def train(self):
        kfold_env = KFoldEnvironment(
            dataset=self.dataset,
            apply_resampling=self.apply_resampling,
            resample_eq_ratio=self.resampling_eq_ratio,
            resample_while_keeping_total_waveforms_fixed=self.resample_while_keeping_total_waveforms_fixed,
        )

        (
            train_gen,
            validation_gen,
            __,
            __,
        ) = kfold_env.get_generators(self.split)

        makedirs(
            get_checkpoint_dir(
                self.exp_name, self.model_name, self.dataset, self.split
            ),
            exist_ok=True,
        )

        model = self._create_model()

        fit_result = self._train_model(
            model=model,
            split=self.split,
            train_gen=train_gen,
            validation_gen=validation_gen,
        )

        self._save_history(self.split, fit_result)

    def _train_model(self, model, split, train_gen, validation_gen):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            eps=self.epsilon,
        )

        history = {"loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            epoch_loss = self._train_one_epoch(model, optimizer, train_gen)
            val_loss = self._validate(model, validation_gen)

            checkpoint_path = get_checkpoint_path(
                self.exp_name, self.model_name, self.dataset, split, epoch
            )
            torch.save(model.state_dict(), checkpoint_path)

            history["loss"].append(float(epoch_loss))
            history["val_loss"].append(float(val_loss))

            print(f"Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        return history

    def _train_one_epoch(self, model, optimizer, train_gen):
        model.train()
        n_batches = len(train_gen)
        epoch_losses = []

        for batch_idx in range(n_batches):
            x_batch, _ = train_gen[batch_idx]
            x_train = torch.from_numpy(x_batch).float().to(self.device)

            optimizer.zero_grad()
            out = model(x_train)
            loss = out[-1]
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss))

            print(f"Batch {batch_idx + 1}/{n_batches} Loss: {float(loss):.4f}")

        return np.mean(epoch_losses)

    def _validate(self, model, validation_gen):
        model.eval()
        n_batches = len(validation_gen)
        val_losses = []

        with torch.no_grad():
            for batch_idx in range(n_batches):
                x_batch, _ = validation_gen[batch_idx]
                x = torch.from_numpy(x_batch).float().to(self.device)

                out = model(x)
                loss = out[-1]
                val_losses.append(float(loss))

        return np.mean(val_losses)

    def _create_model(self):
        model = self.model_class()
        model = model.to(self.device)
        return model

    def _save_history(self, split, fit_result):
        with open(
            get_history_csv_path(self.exp_name, self.model_name, self.dataset, split),
            "w",
        ) as f:
            hist_df = pd.DataFrame(fit_result)
            hist_df.to_csv(f)
