from PIL import Image, ImageEnhance

def sharpness(image: Image.Image, factor: float) -> Image.Image:
    """
    Adjusts the sharpness of the image.
    :param image: A PIL Image object.
    :param factor: A floating point value controlling the enhancement.
                   Factor 1.0 always returns a copy of the original image,
                   lower factors mean less sharp, and higher values more sharp.
    :return: A sharpness-adjusted PIL Image object.
    """
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)
