from PIL import Image
from typing import Tuple

def resize(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """
    Resizes the image to a given size.
    :param image: A PIL Image object.
    :param size: A tuple (width, height) for the new size.
    :return: A resized PIL Image object.
    """
    return image.resize(size)
