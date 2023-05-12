"""Data generation"""
from typing import Callable, Tuple

import numpy as np


def get_data_gen(name: str, im_size: Tuple[int, int], n_samples: int, rgb: bool) -> Callable:
    """Return data generator"""

    def generate_data():
        data = None
        if name == 'flags':
            h, w = im_size
            flag_list = list()
            for sample in range(n_samples):
                l1, l2 = np.sort(np.random.choice(range(w), size=2, replace=False))
                if rgb:
                    r1, r2, r3 = np.random.choice(np.arange(256, dtype=np.uint8), size=3, replace=False)
                    g1, g2, g3 = np.random.choice(np.arange(256, dtype=np.uint8), size=3, replace=False)
                    b1, b2, b3 = np.random.choice(np.arange(256, dtype=np.uint8), size=3, replace=False)
                    # Create the flag
                    flag = np.zeros((h, w, 3), dtype=np.uint8)
                    flag[:, :l1, :] = np.asarray([r1, g1, b1])
                    flag[:, l1:l2] = np.asarray([r2, g2, b2])
                    flag[:, l2:] = np.asarray([r3, g3, b3])
                else:
                    c1, c2, c3 = np.random.choice(np.arange(256, dtype=np.uint8), size=3, replace=False)
                    # Create the flag
                    flag = np.zeros((h, w), dtype=np.uint8)
                    flag[:, :l1] = c1
                    flag[:, l1:l2] = c2
                    flag[:, l2:] = c3

                flag_list.append(flag)  # Add an axis

            flags = np.concatenate([flag_list], axis=0)
            data = flags
        return data

    return generate_data
