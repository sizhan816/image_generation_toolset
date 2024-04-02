import os
import streamlit as st
import torch
import datetime
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

st.set_page_config(
    page_title="通用图生图",
    page_icon="🖼️",
    layout='wide',
    initial_sidebar_state='expanded',
)

DEFAULT_PROMPT = '''
'''.strip()

DEFAULT_NEGATIVE_PROMPT = '''
(low quality:1.3), (worst quality:1.3)
'''.strip()

# Add your custom text here, with smaller font size
st.markdown(
    "<B> 🖼️通用图生图，根据源图生成新图</B>",
    unsafe_allow_html=True)

model_files = os.listdir('./models')

with st.sidebar:
    base_model = st.selectbox(
        "基底模型",
        model_files,
        0,
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
prompt = prompt_cols[0].text_area(
    label="输入提示词",
    height=80,
    value=DEFAULT_PROMPT,
)

negative_prompt = prompt_cols[1].text_area(
    label="输入反向提示词",
    height=80,
    value=DEFAULT_NEGATIVE_PROMPT,
)
uploaded_file = st.file_uploader("上传源图文件")

#@st.cache_data(ttl=3600)  
def get_pipeline():
    pipe = AutoPipelineForImage2Image.from_single_file(
        f"models/{base_model}",
        torch_dtype = torch.float16,
        original_config_file="configs",
        safety_checker = None
    )
    print(use_cuda)
    if use_cuda:
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    if lora_path != "":
        pipe.load_lora_weights(lora_path)

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

    width = int(resize_scale * input_image.width)
    height = int(resize_scale * input_image.height)
    # image 2 image generation
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength = denoising_strength,
        image = input_image,
        generator=generator
    ).images[0]

    # save result image
    os.makedirs(f".\\outputs\\img2img\\{datetime.datetime.now().strftime('%Y-%m-%d')}", exist_ok=True)
    output_path = f".\\outputs\\img2img\\{datetime.datetime.now().strftime('%Y-%m-%d')}\\{datetime.datetime.now().strftime('%H-%M-%S')}_{seed}.png"
    image.save(output_path)

    display_image = Image.open(output_path)

    result_cols = st.columns(2)
    display_width = min(int(resize_scale * input_image.width), 512)
    result_cols[0].image(input_image, caption='源图片', width=display_width)
    result_cols[1].image(display_image, caption='生成的图片', width=display_width)


cols = st.columns(6)
gen_button = cols[5].button("生成", use_container_width=True)
if gen_button:
    generate_img()

