from matplotlib import colors
import numpy as np
import seaborn as sns

def shift_value(rgb, shift):
    hsv = colors.rgb_to_hsv(rgb)
    hsv[-1] += shift
    return colors.hsv_to_rgb(hsv)

def color_palette(n_colors):
    orig_palette = sns.color_palette(n_colors=n_colors)
    shifts = np.linspace(-.2, .2, n_colors)
    alternate_shifts = shifts.copy()
    alternate_shifts[::2] = shifts[:len(shifts[::2])]
    alternate_shifts[1::2] = shifts[len(shifts[::2]):]
    palette = [shift_value(col, shift)
               for col, shift in zip(orig_palette, alternate_shifts)]
    return palette