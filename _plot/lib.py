import os
import matplotlib.pyplot as plt


def set_matplotlib_rc_params():
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "serif",
            "font.serif": "EB Garamond",
            "mathtext.fontset": "cm",
            "text.usetex": False,
            "lines.linewidth": 1.5,
        }
    )


def pages_dir():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(this_dir)
    return os.path.join(root_dir, "pages")


set_matplotlib_rc_params()
