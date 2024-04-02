import os
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import datetime
from PIL import Image


st.set_page_config(
    page_title="通用文生图",
    page_icon="	🅰️",
    layout='wide',
    initial_sidebar_state='expanded',
)

DEFAULT_PROMPT = '''
(cartoon),(best quality),(ultra-detailed),1boy, full body, chibi, yellow, outdoors, beret
'''.strip()

DEFAULT_NEGATIVE_PROMPT = '''
(low quality:1.3), (worst quality:1.3)
'''.strip()

# Add your custom text here, with smaller font size
st.markdown(
    "<B> 🅰️通用文生图，由文字描述生成图片, 支持中文描述自动翻译和根据主体扩写，比原生SD更贴心💖</B>",
    unsafe_allow_html=True)
do_translate = st.checkbox('启用提示词翻译和扩写', value = False)

model_files = os.listdir('./models')

with st.sidebar:
    base_model = st.selectbox(
        "基底模型",
        model_files,
        0,
    )
    lora_path = st.text_input('lora路径:')
    use_cuda = st.checkbox('启用cuda', value = True)
    seed = st.sidebar.number_input(
            "图像生成种子", value=23, min_value=0, max_value=int(1e9)
    )
    width = st.sidebar.number_input(
        "图像宽度:", value=512, min_value=64, max_value=2048
    )
    height = st.sidebar.number_input(
        "图像高度:", value=512, min_value=64, max_value=2048
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
        width = 512
        height = 512
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


#@st.cache_data(ttl=3600)  
def get_pipeline():
    pipe = StableDiffusionPipeline.from_single_file(
        f"models/{base_model}",
        torch_dtype = torch.float16,
        original_config_file="configs/v2-inference.yaml",
        safety_checker = None
     )

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
    
    translated_prompt, translated_negative_prompt = prompt, negative_prompt
    if do_translate:
        from prompt_translate import translator
        translated_prompt = translator.translate_img_generation_prompt(prompt)
        translated_negative_prompt = translator.translate_img_generation_negative_prompt(negative_prompt)
    
    # text 2 image generation
    image = pipe(
        prompt=translated_prompt, 
        negative_prompt=translated_negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]

    # save result image
    os.makedirs(f".\\outputs\\text2img\\{datetime.datetime.now().strftime('%Y-%m-%d')}", exist_ok=True)
    output_path = f".\\outputs\\text2img\\{datetime.datetime.now().strftime('%Y-%m-%d')}\\{datetime.datetime.now().strftime('%H-%M-%S')}_{seed}.png"
    image.save(output_path)

    display_area = st.columns(2)
    display_image = Image.open(output_path)
    display_area[0].image(display_image, caption='生成的图片', width=min(width, 512))
    if do_translate:
        display_area[1].success(f"翻译和扩写后的提示词: {translated_prompt}")
        display_area[1].success(f"翻译后的反向提示词: {translated_negative_prompt}")
        
    

gen_btn_cols = st.columns(6)
# export_btn0, export_btn1, export_btn2, export_btn3, export_btn4 = cols[0], cols[1], cols[2], cols[3], cols[4]
gen_button = gen_btn_cols[5].button("生成", use_container_width=True)
if gen_button:
    generate_img()


