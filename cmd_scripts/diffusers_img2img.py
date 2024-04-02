import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

import argparse

    
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-bm", "--base-model", type=str,
                    help="图生图基底模型名", default="emilianJR/chilloutmix_NiPrunedFp32Fix")
parser.add_argument("-lp", "--lora-path", type=str,
                    help="图生图lora路径", default="N/A")
parser.add_argument("-cu", "--cuda", type=str,
                    help="启用cuda", default=True)

parser.add_argument("-s", "--seed", type=int,
                    help="图生图种子", default=-1)

parser.add_argument("-iip", "--input-image-path", type=str,
                    help="源图路径", required=True)
parser.add_argument("-pr", "--prompt", type=str,
                    help="图生图提示词",
                    default="")
parser.add_argument("-npr", "--negative-prompt", type=str,
                    help="图生图反向提示词",
                    default="(low quality:1.3), (worst quality:1.3)")
parser.add_argument("-st", "--denoising-strength", type=float,
                    help="重绘程度",
                    default=0.5)
parser.add_argument("-wi", "--width", type=int,
                    help="生成图像宽度",
                    default=512)
parser.add_argument("-hi", "--height", type=int,
                    help="生成图像高度",
                    default=768)
parser.add_argument("-gs", "--guidance-scale", type=float,
                    help="指导比例因子",
                    default=9.0)
parser.add_argument("-nis", "--num-inference-steps", type=int,
                    help="迭代轮数",
                    default=32)
parser.add_argument("-o", "--output-path", type=str,
                    help="生成图像路径",
                    default="N/A")

args = parser.parse_args()
print(args)

# text 2 image param setting
base_model = args.base_model
lora_path = args.lora_path
use_cuda = args.cuda
seed = args.seed
i2i_prompt = args.prompt
i2i_negative_prompt = args.negative_prompt
i2i_denoising_strength = args.denoising_strength
img_width = args.width
img_height = args.height
i2i_guidance_scale = args.guidance_scale
i2i_num_inference_steps = args.num_inference_steps
output_path = args.output_path

if output_path == "N/A":
    output_path = f"./out-{seed}.png"


pipe = AutoPipelineForImage2Image.from_pretrained(
    base_model,
    torch_dtype = torch.float16,
    safety_checker = None
)

if use_cuda:
    pipe = pipe.to("cuda")
else:
    pipe = pipe.to("cpu")

if lora_path != "N/A":
    pipe.load_lora_weights(lora_path)

generator = None
if use_cuda:
    generator = torch.Generator("cuda").manual_seed(seed)
else:
    generator = torch.Generator("cpu").manual_seed(seed)

input_image_path = args.input_image_path
input_image = load_image(input_image_path)

# text 2 image generation
image = pipe(
    prompt=i2i_prompt, 
    negative_prompt=i2i_negative_prompt,
    width=img_width,
    height=img_height,
    guidance_scale=i2i_guidance_scale,
    num_inference_steps=i2i_num_inference_steps,
    strength = i2i_denoising_strength,
    image = input_image,
    generator=generator
).images[0]

images = make_image_grid([input_image, image], rows=1, cols=2)
# save result image
image.save(output_path)
images.save(output_path.replace(".","_") + "_grid.png")
