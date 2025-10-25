from PIL import Image, ImageEnhance

def saturation(image: Image.Image, factor: float) -> Image.Image:
    """
    Adjusts the saturation of the image.
    :param image: A PIL Image object.
    :param factor: A floating point value controlling the enhancement.
                   Factor 1.0 always returns a copy of the original image,
                   lower factors mean less color (0.0 is black and white),
                   and higher values more color.
    :return: A saturation-adjusted PIL Image object.
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)
