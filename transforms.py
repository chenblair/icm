import numpy as np
from PIL import Image
import pilgram

def insta_filter(filter_type, amount=None):

    filter_types = ['contrast', 'hue_rotate', 'saturate', 'sepia']

    def transform(img_np, filter_type=filter_type, amount=amount):
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        if (amount is not None):
            transformed = eval(f'pilgram.css.{filter_type}(img, amount)'
            )
        else:
            transformed = eval(f'pilgram.css.{filter_type}(img)')
        return np.array(transformed).astype(np.float64) / 255
    return transform

def randomly_color(arr):
    """Converts grayscale image to random color"""
    color = np.random.randint(0, 3)
    color_arr = [np.zeros(arr.shape) for i in range(3)]
    color_arr[color] = arr
    return np.concatenate(color_arr, axis=1)
