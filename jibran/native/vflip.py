from PIL import Image

def vflip(image: Image.Image) -> Image.Image:
    """
    Vertically flips the image.
    :param image: A PIL Image object.
    :return: A vertically flipped PIL Image object.
    """
    return image.transpose(Image.FLIP_TOP_BOTTOM)
