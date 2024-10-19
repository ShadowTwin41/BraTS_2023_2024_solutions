from monai.transforms.transform import MapTransform
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from collections.abc import Callable, Hashable, Mapping
from monai.config import DtypeLike
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype


import torch
import numpy as np
import time

####
## Transform to scale intensity, following the original WDM 3D github implementation
####
class QuantileAndScaleIntensity(Transform):
    """
    Apply range scaling to a numpy array based on the intensity distribution of the input.

    Args:
        lower: lower quantile.
        upper: upper quantile.
        a_min: intensity target range min.
        a_max: intensity target range max.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    def __init__(self) -> None:
        pass

    def _normalize(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        normalize=(lambda x: 2*x - 1)
        out_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
        out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
        out_normalized= normalize(out_normalized)
        #img = convert_to_tensor(out_normalized, track_meta=False)
        return out_normalized

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        out_normalized = self._normalize(img=img)
        out = convert_to_dst_type(out_normalized, dst=img)[0]
        return out

class QuantileAndScaleIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.QuantileAndScaleIntensity`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        lower: lower quantile.
        upper: upper quantile.
        a_min: intensity target range min.
        a_max: intensity target range max.
        relative: whether to scale to the corresponding percentiles of [a_min, a_max]
        channel_wise: if True, compute intensity percentile and normalize every channel separately.
            default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = QuantileAndScaleIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys=False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = QuantileAndScaleIntensity()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d

