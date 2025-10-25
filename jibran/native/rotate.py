from PIL import Image

def rotate(image: Image.Image, degrees: float) -> Image.Image:
    """
    Rotates the image by a certain number of degrees.
    :param image: A PIL Image object.
    :param degrees: The number of degrees to rotate the image.
    :return: A rotated PIL Image object.
    """
    return image.rotate(degrees, expand=True)
