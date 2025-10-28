import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel,QwenImageEditPlusPipeline
from diffusers.utils import load_image

from PIL import Image
import os 

model_id = "Qwen/Qwen-Image-Edit-2509"
torch_dtype = torch.bfloat16
device = "cuda"

quantization_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
)

transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
transformer = transformer.to("cpu")

quantization_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
text_encoder = text_encoder.to("cpu")

pipe = QwenImageEditPlusPipeline.from_pretrained(
    model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)

#optionally load LoRA weights to speed up inference
pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors")
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
)
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(21)


image1 = Image.open("murdock.jpeg")
image3 = Image.open("boat.png")

prompt = "The man is in the boat"
inputs = {
    "image": [image1, image3],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipe(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")
    print("image saved at", os.path.abspath("output_image_edit_boat.png"))