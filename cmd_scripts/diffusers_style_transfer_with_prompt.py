from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import torch
import cv2
import numpy as np

import argparse

    
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-bm", "--base-model", type=str,
                    help="图生图基底模型名", default="emilianJR/chilloutmix_NiPrunedFp32Fix")
parser.add_argument("-cu", "--cuda", type=str,
                    help="启用cuda", default=True)

parser.add_argument("-iip", "--input-image-path", type=str,
                    help="源图路径", required=True)
parser.add_argument("-spr", "--style-prompt", type=str,
                    help="风格提示词",
                    default="")
parser.add_argument("-npr", "--negative-prompt", type=str,
                    help="反向提示词",
                    default="(low quality:1.3), (worst quality:1.3)")
parser.add_argument("-st", "--denoising-strength", type=float,
                    help="重绘程度",
                    default=0.5)
parser.add_argument("-wi", "--width", type=int,
                    help="生成图像宽度",
                    default=512)
parser.add_argument("-hi", "--height", type=int,
                    help="生成图像高度",
                    default=512)
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

original_image = load_image(args.input_image_path)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
                        args.base_model,
                        controlnet=controlnet,
                        torch_dtype=torch.float16,
                        safety_checker = None
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
if args.cuda:
    pipe = pipe.to("cuda")
else:
    pipe = pipe.to("cpu")

output = pipe(
    "the mona lisa", image=canny_image
).images[0]

output_image = pipe(
    prompt=args.style_prompt, 
    negative_prompt=args.negative_prompt,
    width=args.width,
    height=args.height,
    guidance_scale=args.guidance_scale,
    num_inference_steps=args.num_inference_steps,
    strength = args.denoising_strength,
    image = canny_image
    #generator=generator
).images[0]

images = make_image_grid([original_image, canny_image, output_image], rows=1, cols=3)
images.save(args.output_path)
