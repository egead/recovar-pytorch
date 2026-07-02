from kfold_tester import KFoldTester
from seisbench_kfold_environment import SeisBenchKFoldEnvironment
from seisbench_kfold_trainer import load_seisbench_datasets


class SeisBenchKFoldTester(KFoldTester):
    def __init__(self, *args, event_dataset=None, noise_dataset=None, **kwargs):
        if event_dataset is None or noise_dataset is None:
            event_dataset, noise_dataset = load_seisbench_datasets(kwargs["test_dataset"])
        self.event_dataset = event_dataset
        self.noise_dataset = noise_dataset
        super().__init__(*args, **kwargs)

    def _add_test_environment(self):
        self.test_environment = SeisBenchKFoldEnvironment(
            self.test_dataset,
            event_dataset=self.event_dataset,
            noise_dataset=self.noise_dataset,
            apply_resampling=self.apply_resampling,
            resample_eq_ratio=self.resample_eq_ratio,
            resample_while_keeping_total_waveforms_fixed=self.resample_while_keeping_total_waveforms_fixed,
        )
