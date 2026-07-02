from evaluator import Evaluator, CropOffsetFilter, SNRFilter, TracesFilter
from seisbench_kfold_tester import SeisBenchKFoldTester


class SeisBenchEvaluator(Evaluator):
    def _add_tester(self, representation_learning_model_class, classifier_model_class):
        self.tester = SeisBenchKFoldTester(
            self.exp_name,
            representation_learning_model_class=representation_learning_model_class,
            classifier_model_class=classifier_model_class,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            split=self.split,
            epochs=self.epochs,
            resample_while_keeping_total_waveforms_fixed=self.resample_while_keeping_total_waveforms_fixed,
            method_params=self.method_params,
            apply_resampling=self.apply_resampling,
            resample_eq_ratio=self.resample_eq_ratio,
        )
