from enum import Enum


class ClassNames(str, Enum):
    NO_PLUME = "no_plume"
    PLUME = "plume"


CLASS_LABEL_MAP = {
    ClassNames.NO_PLUME.value: 0,
    ClassNames.PLUME.value: 1
}

LABEL_CLASS_MAP = {
    0: ClassNames.NO_PLUME.value,
    1: ClassNames.PLUME.value
}
