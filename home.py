import streamlit as st

st.set_page_config(
    page_title="SD 工具箱",
    page_icon="🏠",
    layout='wide',
    initial_sidebar_state='expanded',
)

st.write("# 欢迎来到SD工具箱 📦")

st.sidebar.success("选择您想运行的demo")
st.markdown(
    """
    SD 工具箱是一系列基于开源[diffusers](https://huggingface.co/docs/diffusers/index)开发的场景化工具集,
    其底层算法为[stable diffusion](https://github.com/CompVis/stable-diffusion)。
    
    当前我们支持以下应用场景：
    - 通用文生图：输入一段文本，生成一张图片，支持中文描述自动翻译和根据主体扩写，比原生SD更贴心
    - 通用图生图：基于源图片生成新图片
    - 图像风格迁移：基于源图片和风格迁移描述生成转换风格的新图片
    
    **👈 从左侧工具集中选择您要体验的demo
    未来我们将支持更多场景包括但不限于：
    - 聊天生图: 引导用户通过多轮聊天的方式生成和修改图片
    - 更多一键生图工具，例如：一键换装、一键脱衣、一键换脸、场景变换
    - 视频生成和修改工具，例如：文生视频、照片生视频、视频风格迁移、视频换脸、视频换装
    - 商用工具，例如：logo设计、工艺品设计、室内设计、广告创意
    - 趣味案例，例如：人物长相融合、艺术字照片、回到童年
    
    ** 💬合作、咨询、拜师, 认准思南、纲波、凯龙，🌍：chelloworldu
"""
)

