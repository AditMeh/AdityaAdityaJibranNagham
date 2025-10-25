from PIL import Image

def grayscale(image: Image.Image) -> Image.Image:
    """
    Converts the image to grayscale.
    :param image: A PIL Image object.
    :return: A grayscaled PIL Image object.
    """
    return image.convert('L')
