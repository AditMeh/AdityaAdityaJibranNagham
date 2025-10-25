from PIL import Image
import os

# Import all the native functions
from hflip import hflip
from vflip import vflip
from grayscale import grayscale
from rotate import rotate
from resize import resize
from sharpness import sharpness
from saturation import saturation

def run_demonstration():
    """
    Loads an image and demonstrates each of the native image transformation functions.
    The output of each transformation is saved to a new file and displayed.
    """
        # Load the original image
    original_image = Image.open("example.jpg")
    print("Original image loaded.")

    # Create an output directory if it doesn't exist
    output_dir = "jibran/native/test_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Grayscale ---
    print("Applying grayscale...")
    grayscale_image = grayscale(original_image)
    grayscale_image.save(os.path.join(output_dir, "example_grayscale.jpg"))
    grayscale_image.show()

    # --- 2. Horizontal Flip ---
    print("Applying horizontal flip...")
    hflip_image = hflip(original_image)
    hflip_image.save(os.path.join(output_dir, "example_hflip.jpg"))
    hflip_image.show()

    # --- 3. Vertical Flip ---
    print("Applying vertical flip...")
    vflip_image = vflip(original_image)
    vflip_image.save(os.path.join(output_dir, "example_vflip.jpg"))
    vflip_image.show()

    # --- 4. Rotate ---
    print("Applying 90-degree rotation...")
    rotate_image = rotate(original_image, 90)
    rotate_image.save(os.path.join(output_dir, "example_rotate_90.jpg"))
    rotate_image.show()

    # --- 5. Resize ---
    print("Applying resize to (300, 300)...")
    resize_image = resize(original_image, (300, 300))
    resize_image.save(os.path.join(output_dir, "example_resize_300x300.jpg"))
    resize_image.show()

    # --- 6. Sharpness ---
    print("Applying sharpness enhancement (factor 2.0)...")
    sharpness_image = sharpness(original_image, 2.0)
    sharpness_image.save(os.path.join(output_dir, "example_sharpness_2.0.jpg"))
    sharpness_image.show()
    
    # --- 7. Saturation ---
    print("Applying saturation enhancement (factor 2.0)...")
    saturation_image = saturation(original_image, 2.0)
    saturation_image.save(os.path.join(output_dir, "example_saturation_2.0.jpg"))
    saturation_image.show()

    print(f"\nAll transformed images have been saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    run_demonstration()