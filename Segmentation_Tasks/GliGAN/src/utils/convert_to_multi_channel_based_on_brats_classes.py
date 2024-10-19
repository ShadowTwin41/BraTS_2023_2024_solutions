import numpy as np
import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import MapTransform
from collections.abc import Callable, Hashable, Mapping


class ConvertToMultiChannelBasedOnBratsGliomaClasses2023(Transform):
    """
    Convert labels to multi channels based on brats23 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    Return:
        The possible classes TC (Tumor core), WT (Whole tumor) and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 1) | (img == 3), (img == 1) | (img == 2) | (img == 3), img == 3]
        # merge labels 1 (tumor non-enh) and 3 (tumor enh) and 2 (large edema) to WT
        # label 3 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

class ConvertToMultiChannelBasedOnBratsGliomaClasses2023d(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsGliomaClasses2023`.
    Convert labels to multi channels based on brats23 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    Return:
        The possible classes TC (Tumor core), WT (Whole tumor) and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnBratsGliomaClasses2023.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsGliomaClasses2023()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

class ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024(Transform):
    """
    Convert labels to multi channels based on brats24 Glioma post treatment classes:

    label 1: NETC denotes necrosis and cysts within the tumor.
    label 2: SNFH includes edema, infiltrating tumor, and post-treatment changes.
    label 3: ET describes the regions of active tumor as well as nodular areas of enhancement.
    label 4: RC consists of both recent and chronic resection cavities and typically contains fluid, blood, air, and/or proteinaceous materials.
    Return:
        The possible classes TC (Tumor core), WT (Whole tumor), ET (Enhancing tumor) and RC (resection cavity).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 1) | (img == 3), (img == 1) | (img == 2) | (img == 3), img == 3, img == 4]
        # Tumor core (ET plus NETC) describes what is typically resected during a surgical procedure.
        # Whole tumor (ET plus SNFH plus NETC) defines the whole extent of the tumor, including the tumor core, infiltrating tumor, peritumoral edema and treatment-related changes.
        # ET describes the regions of active tumor as well as nodular areas of enhancement.
        # RC consists of both recent and chronic resection cavities and typically contains fluid, blood, air, and/or proteinaceous materials.
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

class ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024`.
     Convert labels to multi channels based on brats24 Glioma post treatment classes:

    label 1: NETC denotes necrosis and cysts within the tumor.
    label 2: SNFH includes edema, infiltrating tumor, and post-treatment changes.
    label 3: ET describes the regions of active tumor as well as nodular areas of enhancement.
    label 4: RC consists of both recent and chronic resection cavities and typically contains fluid, blood, air, and/or proteinaceous materials.
    Return:
        The possible classes TC (Tumor core), WT (Whole tumor), ET (Enhancing tumor) and RC (resection cavity).
    """
    backend = ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

class ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024(Transform):
    """
    Convert labels to multi channels based on brats24 meningioma classe:

    label 1: Gross Tumor Volume (GTV)
   
    Return:
        The possible classes GTV.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [img == 1]
        # Meningioma Radiotherapy Gross Tumor Volume (GTV)
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

class ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024d(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024`.
    Convert labels to multi channels based on brats24 meningioma classe:

    label 1: Gross Tumor Volume (GTV)
   
    Return:
        The possible classes GTV.
    """

    backend = ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


