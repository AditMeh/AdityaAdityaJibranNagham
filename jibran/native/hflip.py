from PIL import Image

def hflip(image: Image.Image) -> Image.Image:
    """
    Horizontally flips the image.
    :param image: A PIL Image object.
    :return: A horizontally flipped PIL Image object.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)
