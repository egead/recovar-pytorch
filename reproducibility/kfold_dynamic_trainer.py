import numpy as np
import pandas as pd
import torch
from recovar_torch import BATCH_SIZE, ClassifierMultipleAutoencoder
from recovar_torch.utils import demean, l2_normalize
from kfold_trainer import KfoldTrainer
from kfold_environment import KFoldEnvironment
from directory import *
from os import makedirs


class KfoldDynamicTrainer(KfoldTrainer):
    def __init__(
        self,
        exp_name,
        model_class,
        dataset,
        split,
        epochs,
        final_dilation_number=64,
        apply_resampling=False,
        resampling_eq_ratio=0.5,
        resample_while_keeping_total_waveforms_fixed=False,
        learning_rate=1e-4,
        epsilon=1e-7,
        beta_1=0.99,
        beta_2=0.999,
    ):
        super().__init__(
            exp_name=exp_name,
            model_class=model_class,
            dataset=dataset,
            split=split,
            epochs=epochs,
            apply_resampling=apply_resampling,
            resampling_eq_ratio=resampling_eq_ratio,
            resample_while_keeping_total_waveforms_fixed=resample_while_keeping_total_waveforms_fixed,
            learning_rate=learning_rate,
            epsilon=epsilon,
            beta_1=beta_1,
            beta_2=beta_2,
        )

        self.final_dilation_number = final_dilation_number

    def train(self):
        kfold_env = KFoldEnvironment(
            dataset=self.dataset,
            apply_resampling=self.apply_resampling,
            resample_eq_ratio=self.resampling_eq_ratio,
            resample_while_keeping_total_waveforms_fixed=self.resample_while_keeping_total_waveforms_fixed,
        )

        train_gen, validation_gen, _, _ = kfold_env.get_generators(self.split)

        makedirs(
            get_checkpoint_dir(
                self.exp_name, self.model_name, self.dataset, self.split
            ),
            exist_ok=True,
        )

        model = self._create_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            eps=self.epsilon,
        )

        classifier = ClassifierMultipleAutoencoder(model=model)

        history = {"loss": [], "val_loss": []}

        for epoch in range(self.epochs):

            epoch_loss = self._train_one_epoch(
                model, optimizer, train_gen, classifier, epoch
            )
            val_loss = self._validate(model, validation_gen)

            checkpoint_path = get_checkpoint_path(
                self.exp_name, self.model_name, self.dataset, self.split, epoch
            )
            torch.save(model.state_dict(), checkpoint_path)

            history["loss"].append(float(epoch_loss))
            history["val_loss"].append(float(val_loss))

            print(f"Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        self._save_history(self.split, history)

    def _train_step(self, model, optimizer, x_train):
        model.train()
        x_train = torch.from_numpy(x_train).float().to(self.device)

        out = model(x_train)
        f1, f2, f3, f4, f5 = out[0], out[1], out[2], out[3], out[4]
        y1, y2, y3, y4, y5 = out[5], out[6], out[7], out[8], out[9]

        per_sample_recon_loss = (
            self._per_sample_l2_distance(x_train, y1)
            + self._per_sample_l2_distance(x_train, y2)
            + self._per_sample_l2_distance(x_train, y3)
            + self._per_sample_l2_distance(x_train, y4)
            + self._per_sample_l2_distance(x_train, y5)
        ) / 5.0

        per_sample_ensemble_loss = (
            self._per_sample_ensemble_distance(f1, f2)
            + self._per_sample_ensemble_distance(f1, f3)
            + self._per_sample_ensemble_distance(f2, f3)
            + self._per_sample_ensemble_distance(f1, f4)
            + self._per_sample_ensemble_distance(f2, f4)
            + self._per_sample_ensemble_distance(f3, f4)
            + self._per_sample_ensemble_distance(f1, f5)
            + self._per_sample_ensemble_distance(f2, f5)
            + self._per_sample_ensemble_distance(f3, f5)
            + self._per_sample_ensemble_distance(f4, f5)
        ) / 10.0

        per_sample_total_loss = per_sample_recon_loss + per_sample_ensemble_loss
        loss = torch.mean(per_sample_total_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss)

    def _get_dilation_number(self, epoch):
        dilation_numbers = np.logspace(np.log2(1.0), np.log2(self.final_dilation_number), base=2, num=self.epochs)
        return np.round(dilation_numbers[epoch]).astype(np.int32)

    def _get_dilated_batch(self, train_gen, batch_iterator, dilation, classifier):
        if dilation <= 1:
            x_batch, _ = train_gen[batch_iterator]
            return x_batch, batch_iterator + 1

        grouped_batches = []
        grouped_batch_scores = []

        n_batches = len(train_gen)

        for _ in range(dilation):
            if batch_iterator >= n_batches:
                break

            x_batch, _ = train_gen[batch_iterator]

            classifier.eval()
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_batch).float().to(self.device)
                batch_scores = classifier(x_tensor).cpu().numpy()

            grouped_batches.append(x_batch)
            grouped_batch_scores.append(batch_scores)
            batch_iterator += 1

        x_pool = np.concatenate(grouped_batches, axis=0)
        scores = np.concatenate(grouped_batch_scores, axis=0)

        kept_idxs = np.argsort(scores)[::-1][:BATCH_SIZE]
        x_selected = x_pool[kept_idxs]

        return x_selected, batch_iterator

    def _train_one_epoch(self, model, optimizer, train_gen, classifier, epoch):
        dilation = self._get_dilation_number(epoch)
        n_batches = len(train_gen)
        epoch_losses = []
        batch_iterator = 0

        while batch_iterator < n_batches:
            x_batch, batch_iterator = self._get_dilated_batch(
                train_gen, batch_iterator, dilation, classifier
            )

            loss = self._train_step(model, optimizer, x_batch)
            epoch_losses.append(float(loss))

            print(f"Batch {batch_iterator}/{n_batches} Loss: {loss:.4f}")

        return np.mean(epoch_losses)

    def _validate(self, model, validation_gen):
        model.eval()
        n_batches = len(validation_gen)
        val_losses = []

        with torch.no_grad():
            for batch_idx in range(n_batches):
                x, y = validation_gen[batch_idx]
                x = torch.from_numpy(x).float().to(self.device)

                out = model(x)
                f1, f2, f3, f4, f5 = out[0], out[1], out[2], out[3], out[4]
                y1, y2, y3, y4, y5 = out[5], out[6], out[7], out[8], out[9]

                per_sample_recon_loss = (
                    self._per_sample_l2_distance(x, y1)
                    + self._per_sample_l2_distance(x, y2)
                    + self._per_sample_l2_distance(x, y3)
                    + self._per_sample_l2_distance(x, y4)
                    + self._per_sample_l2_distance(x, y5)
                ) / 5.0

                per_sample_ensemble_loss = (
                    self._per_sample_ensemble_distance(f1, f2)
                    + self._per_sample_ensemble_distance(f1, f3)
                    + self._per_sample_ensemble_distance(f2, f3)
                    + self._per_sample_ensemble_distance(f1, f4)
                    + self._per_sample_ensemble_distance(f2, f4)
                    + self._per_sample_ensemble_distance(f3, f4)
                    + self._per_sample_ensemble_distance(f1, f5)
                    + self._per_sample_ensemble_distance(f2, f5)
                    + self._per_sample_ensemble_distance(f3, f5)
                    + self._per_sample_ensemble_distance(f4, f5)
                ) / 10.0

                per_sample_total_loss = per_sample_recon_loss + per_sample_ensemble_loss
                loss = torch.mean(per_sample_total_loss)

                val_losses.append(float(loss))

        return np.mean(val_losses)

    def _per_sample_l2_distance(self, x, y):
        x_demeaned = demean(x)
        y_demeaned = demean(y)

        distance = torch.sqrt(torch.mean(torch.square(x_demeaned - y_demeaned), dim=(1, 2)))
        return distance

    def _per_sample_ensemble_distance(self, f1, f2):
        f1_normalized = demean(f1, axis=1)
        f2_normalized = demean(f2, axis=1)

        f1_normalized = l2_normalize(f1_normalized, axis=1)
        f2_normalized = l2_normalize(f2_normalized, axis=1)

        distance = torch.sqrt(torch.mean(torch.square(f1_normalized - f2_normalized), dim=(1, 2)))
        return distance

    def _save_history(self, split, history_dict):
        with open(
            get_history_csv_path(self.exp_name, self.model_name, self.dataset, split),
            "w",
        ) as f:
            hist_df = pd.DataFrame(history_dict)
            hist_df.to_csv(f)
