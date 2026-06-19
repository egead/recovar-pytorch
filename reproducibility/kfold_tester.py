import pandas as pd
import torch
from os import makedirs
from recovar_torch import BATCH_SIZE
from kfold_environment import KFoldEnvironment
import numpy as np
from directory import *

class KFoldTester:
    def __init__(
        self,
        exp_name,
        representation_learning_model_class,
        classifier_model_class,
        train_dataset,
        test_dataset,
        split,
        epochs,
        apply_resampling,
        resample_while_keeping_total_waveforms_fixed,
        resample_eq_ratio,
        method_params={},
    ):
        self.exp_name = exp_name
        self.representation_learning_model_class = representation_learning_model_class
        self.classifer_model_class = classifier_model_class
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.split = split
        self.epochs = epochs
        self.method_params = method_params
        self.resample_while_keeping_total_waveforms_fixed = resample_while_keeping_total_waveforms_fixed
        self.apply_resampling = apply_resampling
        self.resample_eq_ratio = resample_eq_ratio

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.representation_learning_model_name = representation_learning_model_class().name
        self.classifier_model_name = classifier_model_class().name
        self._add_test_environment()

    def test(
        self,
    ):
        exp_results_dir = self._get_exp_results_dir()

        makedirs(exp_results_dir, exist_ok=True)
        __, __, __, predict_gen = self.test_environment.get_generators(self.split)

        for epoch in self.epochs:
            classifier_model = self._create_classifier_model(epoch)
            scores = self._predict(classifier_model, predict_gen)
            self._save_score_file(scores, epoch)

        __, __, metadata = self.test_environment.get_split_metadata(self.split)
        self._save_meta_file(metadata)

    def _predict(self, classifier_model, predict_gen):
        outputs = []
        n_batches = predict_gen.__len__()
        for i in range(n_batches):
            x = predict_gen.__getitem__(i)
            xt = torch.from_numpy(np.asarray(x)).float().to(self.device)
            with torch.no_grad():
                y = classifier_model(xt)
            y = y.cpu().numpy()
            print(f"Prediction completed:{i}/{n_batches}")
            outputs.append(y)

        output = np.concatenate(outputs, axis=0)
        return output

    def _save_score_file(self, scores, epoch):
        score_file_path = get_exp_results_score_file_path(
            self.exp_name,
            self.representation_learning_model_name,
            self.classifier_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
            epoch
        )

        df_score = pd.DataFrame({"eq_probabilities":scores})
        df_score.to_csv(score_file_path)

    def _save_meta_file(self, metadata):
        meta_file_path = get_exp_results_meta_file_path(
            self.exp_name,
            self.representation_learning_model_name,
            self.classifier_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
        )

        metadata.to_csv(meta_file_path)

    def _create_representation_learning_model(self):
        model = self.representation_learning_model_class()
        model.to(self.device)
        model.eval()

        return model

    def _create_classifier_model(self, epoch):
        if self.representation_learning_model_class is not None:
            representation_learning_model = self._create_representation_learning_model()

            representation_learning_model.load_state_dict(
                torch.load(
                    get_checkpoint_path(
                        self.exp_name,
                        self.representation_learning_model_name,
                        self.train_dataset,
                        self.split,
                        epoch,
                    ),
                    map_location=self.device,
                )
            )
        else:
            representation_learning_model = None

        classifier_model = self.classifer_model_class(
            representation_learning_model,
            method_params=self.method_params,
        )
        classifier_model.to(self.device)
        classifier_model.eval()
        return classifier_model

    def _add_test_environment(self):
        self.test_environment = KFoldEnvironment(self.test_dataset,
                                                 apply_resampling=self.apply_resampling,
                                                 resample_eq_ratio=self.resample_eq_ratio,
                                                 resample_while_keeping_total_waveforms_fixed=self.resample_while_keeping_total_waveforms_fixed)

    def _get_exp_results_dir(self):
        return get_exp_results_dir(
            self.exp_name,
            self.representation_learning_model_name,
            self.classifier_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
        )
