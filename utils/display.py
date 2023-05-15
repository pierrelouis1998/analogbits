import numpy as np
from matplotlib import ticker, pyplot as plt


def show_flag(flag: np.ndarray, title = None):
    if not title:
        title = 'A sample of the flag dataset'
    plt.imshow(flag)
    plt.title(title, fontsize=16)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.NullLocator())
    _ = ax.yaxis.set_major_locator(ticker.NullLocator())
