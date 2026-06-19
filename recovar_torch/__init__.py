from recovar_torch.representation_learning_models import (
    AutoencoderBlock,
    RepresentationLearningSingleAutoencoder,
    RepresentationLearningDenoisingSingleAutoencoder,
    RepresentationLearningMultipleAutoencoder,
    RepresentationLearningMultipleAutoencoderL4,
)

from recovar_torch.classifier_models import (
    ClassifierAutocovariance,
    ClassifierAugmentedAutoencoder,
    ClassifierMultipleAutoencoder,
)

from recovar_torch.config import BATCH_SIZE, SAMPLING_FREQ

from recovar_torch.cubic_interpolation import diff, cubic_interp1d

from recovar_torch.utils import demean, l2_distance, l2_normalize, l4_normalize, l4_distance

from recovar_torch.layers import (
    AddNoise,
    NormalizeStd,
    Padding,
    Conv,
    Upsample,
    UpsampleNoactivation,
    Downsample,
    ResIdentity,
    CrossCovarianceCircular,
)

__all__ = [
    "AutoencoderBlock",
    "RepresentationLearningSingleAutoencoder",
    "RepresentationLearningDenoisingSingleAutoencoder",
    "RepresentationLearningMultipleAutoencoder",
    "RepresentationLearningMultipleAutoencoderL4",
    "ClassifierAutocovariance",
    "ClassifierAugmentedAutoencoder",
    "ClassifierMultipleAutoencoder",
    "BATCH_SIZE",
    "SAMPLING_FREQ",
]

__version__ = "0.1.0"
