from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class RandomState:
    seed: int

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
