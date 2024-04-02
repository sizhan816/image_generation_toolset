import streamlit as st
import os
import streamlit as st
import torch
import datetime
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import cv2
import numpy as np

st.set_page_config(
    page_title="风格迁移",
    page_icon="🔃",
    layout='wide',
    initial_sidebar_state='expanded',
)

DEFAULT_PROMPT = '''
    Van Gogh's Starry Night
'''.strip()

DEFAULT_NEGATIVE_PROMPT = '''
(low quality:1.3), (worst quality:1.3)
'''.strip()

# Add your custom text here, with smaller font size
st.markdown(
    "<B> 🖼️风格迁移（提示词驱动），根据源图和风格描述生成新图</B>",
    unsafe_allow_html=True)

model_files = os.listdir('./models')


with st.sidebar:
    base_model = st.selectbox(
        "基底模型",
        model_files,
        0
    )
    lora_path = st.text_input('lora路径:')
    use_cuda = st.checkbox('启用cuda', value=True)
    seed = st.sidebar.number_input(
            "图像生成种子", value=23, min_value=0, max_value=int(1e9)
    )
    denoising_strength = st.sidebar.slider(
        '重绘强度', 0.0, 1.0, 0.8, step=0.01
    )
    resize_scale = st.sidebar.number_input(
        "缩放比例", value=1.0, min_value=0.1, max_value=2.0
    )
    guidance_scale = st.slider(
        '指导因子', 0.0, 20.0, 7.5, step=0.1
    )
    num_inference_steps = st.slider(
        '推理步数', 1, 150, 20, step=1
    )

    cols = st.columns(2)
    export_btn = cols[0]
    reset = cols[1].button("重置参数", use_container_width=True)
    if reset:
        base_model = model_files[0]
        lora_path = ''
        use_cuda = True
        seed = 23
        denoising_strength = 0.8
        resize_scale = 1.0
        guidance_scale = 7.5
        num_inference_steps = 20

prompt_cols = st.columns(2)
style_prompt = prompt_cols[0].text_area(
    label="输入风格描述",
    height=80,
    value=DEFAULT_PROMPT,
)

negative_prompt = prompt_cols[1].text_area(
    label="输入反向提示词",
    height=80,
    value=DEFAULT_NEGATIVE_PROMPT,
)
uploaded_file = st.file_uploader("上传源图文件")

def get_pipeline():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        f"models/{base_model}",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        original_config_file="configs",
        safety_checker = None
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    if use_cuda:
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    
    return pipe

def generate_img():
    pipe = get_pipeline()

    generator = None
    if use_cuda:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = torch.Generator("cpu").manual_seed(seed)

    input_pil_img = Image.open(uploaded_file)
    input_image = load_image(input_pil_img)
    image_np = np.array(input_image)
    low_threshold = 100
    high_threshold = 200
    image_np = cv2.Canny(image_np, low_threshold, high_threshold)
    image_np = image_np[:, :, None]
    image_np = np.concatenate([image_np, image_np, image_np], axis=2)
    canny_image = Image.fromarray(image_np)

    width = int(resize_scale * input_image.width)
    height = int(resize_scale * input_image.height)
    print(f"width:{width}, height:{height}, resize_scale:{resize_scale}")
    # style transfer generation
    output_image = pipe(
        prompt=style_prompt, 
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength = denoising_strength,
        image = canny_image,
        generator=generator
    ).images[0]

    # save result image
    os.makedirs(f".\\outputs\\img2img\\{datetime.datetime.now().strftime('%Y-%m-%d')}", exist_ok=True)
    output_path = f".\\outputs\\img2img\\{datetime.datetime.now().strftime('%Y-%m-%d')}\\{datetime.datetime.now().strftime('%H-%M-%S')}_{seed}.png"
    output_image.save(output_path)
   
    input_image_resized = input_image.resize((output_image.width, output_image.height))
    canny_image_resized = canny_image.resize((output_image.width, output_image.height))
    images_grid = make_image_grid([input_image_resized, canny_image_resized, output_image], rows=1, cols=3)
    grid_output_path = f".\\outputs\\img2img\\{datetime.datetime.now().strftime('%Y-%m-%d')}\\{datetime.datetime.now().strftime('%H-%M-%S')}_{seed}_grid.png"
    images_grid.save(grid_output_path)

    display_image = Image.open(grid_output_path)
    display_width = min(display_image.width, 1080)
    st.image(display_image, caption='从左至右分别为源图片、源图片线条、风格化图片', width=display_width)
    

cols = st.columns(6)
gen_button = cols[5].button("生成", use_container_width=True)
if gen_button:
    generate_img()


