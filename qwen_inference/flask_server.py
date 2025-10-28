from flask import Flask, request, send_file
import torch
import random
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from PIL import Image
import io
import os
import numpy as np
from lang_sam import LangSAM

app = Flask(__name__)

# Model configuration
model_id = "Qwen/Qwen-Image-Edit"
multi_model_id = "Qwen/Qwen-Image-Edit-2509"
torch_dtype = torch.bfloat16
device = "cuda"

# -------- Model Initialization (startup, happens ONCE) --------
def init_models():
    diffusers_quant_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=diffusers_quant_config,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    transformers_quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=transformers_quant_config,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    pipe = QwenImageEditPipeline.from_pretrained(
        model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
    )
    pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors")
    pipe.to(device)
    return pipe

def init_multi_models():
    diffusers_quant_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        multi_model_id,
        subfolder="transformer",
        quantization_config=diffusers_quant_config,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    transformers_quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        multi_model_id,
        subfolder="text_encoder",
        quantization_config=transformers_quant_config,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        multi_model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
    )
    
    # Load LoRA weights to speed up inference
    pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors")
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
    )
    
    # Enable CPU offload to manage memory
    pipe.enable_model_cpu_offload()
    pipe.to(device)
    return pipe

def preprocess_to_uint8(img):
    """
    Converts a NumPy image array to uint8 for saving.
    Handles floats in [0,1] or any numeric range.
    """
    if img.dtype == np.uint8:
        return img  # already fine

    img = np.nan_to_num(img)  # remove NaNs/Infs
    img = img - img.min()     # shift to start at 0
    if img.max() > 0:
        img = img / img.max() # normalize to [0,1]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

def get_bounding_box(mask):
    """
    Get bounding box coordinates from a binary mask.
    Returns (x_min, y_min, x_max, y_max) or None if no mask found.
    """
    # Find where mask is True/non-zero
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return (x_min, y_min, x_max, y_max)

# Initialize models on startup
print("Initializing single image model...")
pipe = init_models()
print("Initializing multi-image model...")
multi_pipe = init_multi_models()
print("Initializing LangSAM model...")
langsam_model = LangSAM()
generator = torch.Generator(device=device).manual_seed(42)
print("All models initialized successfully!")

@app.route('/', methods=['GET'])
def health_check():
    return "Image generation server is running!"

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Check if image and prompt are provided
        if 'image' not in request.files:
            return {"error": "No image provided"}, 400
        
        if 'prompt' not in request.form:
            return {"error": "No prompt provided"}, 400
        
        # Get the uploaded image and prompt
        image_file = request.files['image']
        prompt = request.form['prompt']
        
        if image_file.filename == '':
            return {"error": "No image selected"}, 400
        
        if not prompt.strip():
            return {"error": "Empty prompt"}, 400
        
        # Load and process the image
        image = Image.open(image_file.stream).convert("RGB")
        
        # Run inference
        print(f"Running image generation for: '{prompt}'...")
        output = pipe(image, prompt, num_inference_steps=6, generator=generator)
        generated_image = output.images[0]
        
        # Save to 'output.png'
        output_path = os.path.abspath("output.png")
        generated_image.save(output_path)

        # Return the generated image
        return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='output.png')
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"error": f"Generation failed: {str(e)}"}, 500

@app.route('/generate_multi', methods=['POST'])
def generate_multi_image():
    try:
        # Check if images and prompt are provided
        if 'images' not in request.files:
            return {"error": "No images provided"}, 400
        
        if 'prompt' not in request.form:
            return {"error": "No prompt provided"}, 400
        
        # Get the uploaded images and prompt
        image_files = request.files.getlist('images')
        prompt = request.form['prompt']
        
        if not image_files or all(file.filename == '' for file in image_files):
            return {"error": "No images selected"}, 400
        
        if not prompt.strip():
            return {"error": "Empty prompt"}, 400
        
        # Load and process the images
        images = []
        for image_file in image_files:
            if image_file.filename:  # Skip empty files
                image = Image.open(image_file.stream).convert("RGB")
                images.append(image)
        
        if not images:
            return {"error": "No valid images provided"}, 400
        
        # Run inference
        print(f"Running multi-image generation for: '{prompt}' with {len(images)} images...")
        
        inputs = {
            "image": images,
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": 4.0,
            "negative_prompt": "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        }
        
        with torch.inference_mode():
            output = multi_pipe(**inputs)
            generated_image = output.images[0]
        
        # Save to 'output_multi.png'
        output_path = os.path.abspath("output_multi.png")
        generated_image.save(output_path)

        # Return the generated image
        return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='output_multi.png')
        
    except Exception as e:
        print(f"Error during multi-image generation: {e}")
        return {"error": f"Multi-image generation failed: {str(e)}"}, 500

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        # Check if image and prompt are provided
        if 'image' not in request.files:
            return {"error": "No image provided"}, 400
        
        if 'prompt' not in request.form:
            return {"error": "No prompt provided"}, 400
        
        # Get the uploaded image and prompt
        image_file = request.files['image']
        prompt = request.form['prompt']
        
        if image_file.filename == '':
            return {"error": "No image selected"}, 400
        
        if not prompt.strip():
            return {"error": "Empty prompt"}, 400
        
        # Load and process the image
        image = Image.open(image_file.stream).convert("RGB")
        
        # Run LangSAM inference
        print(f"Running segmentation for: '{prompt}'...")
        results = langsam_model.predict([image], [prompt])[0]["masks"]
        
        # Check if we got any masks
        if len(results) == 0:
            print("No masks found, returning original image")
            # Return original image
            output_path = os.path.abspath("output_segment.png")
            image.save(output_path)
            return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='output_segment.png')
        
        # Get the first mask
        mask = results[0]
        print(f"Mask shape: {mask.shape}")
        
        # Check if mask is grayscale (1 channel or 2D)
        if len(mask.shape) == 2 or mask.shape[2] == 1:
            print("Mask is grayscale, processing...")
            
            # Convert mask to uint8 if needed
            mask_uint8 = preprocess_to_uint8(mask)
            
            # Get bounding box
            bbox = get_bounding_box(mask_uint8)
            
            if bbox is None:
                print("No valid mask found, returning original image")
                # Return original image
                output_path = os.path.abspath("output_segment.png")
                image.save(output_path)
                return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='output_segment.png')
            
            x_min, y_min, x_max, y_max = bbox
            print(f"Cropping to bounding box: ({x_min}, {y_min}, {x_max}, {y_max})")
            
            # Crop the image to the bounding box
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            # Save cropped image
            output_path = os.path.abspath("output_segment.png")
            cropped_image.save(output_path)
            print(f"Saved cropped image to: {output_path}")
            
            # Return the cropped image
            return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='output_segment.png')
        
        else:
            print(f"Mask is not grayscale (shape: {mask.shape}), returning original image")
            # Return original image
            output_path = os.path.abspath("output_segment.png")
            image.save(output_path)
            return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='output_segment.png')
        
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return {"error": f"Segmentation failed: {str(e)}"}, 500

if __name__ == '__main__':
    app.run(host='localhost', port=6000, debug=False)
