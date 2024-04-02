import os
import platform
import torch
from transformers import AutoTokenizer, AutoModel
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

os_name = platform.system()

class ImageGenerationPromptTranslate(object):
    tokenizer: AutoTokenizer
    model: AutoModel 
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True, device=device)
        self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device=device).eval()
        self.translate_img_generation_prompt("女孩包饺子")
    def __call__(self, user_prompt):
        return self._build_prompt(self, user_prompt)
    def _build_prompt(self, user_prompt):
        return f"""
            你是一个AI画图提示词翻译专家，帮助用户把画图需求转换为英文提示词.
            要求:
            1.如果用户需求是包含中文则需要翻译为英文再进行翻译，如果用户提示词是英文则直接翻译即可，不需要进行语种翻译。 
            2.保留用户需求本意，翻译后的英文提示词最好是逗号隔开的描述性短语，根据用户需求加入合适的一些场景、背景、画风、图片质量等扩展描述，一共10-20个短语左右；
            3. 描述性短语由下列几类构成：
            a. 主体词:例如girl, boy, cat, dog, elephant, etc.
            b. 描述词:例如, beautiful, delicate, delicate appearance, vibrant, etc.
            c. 背景词:例如, in a park, in a garden, in a forest, etc.环境词和主体最好比较符合逻辑，如果用户提供的原始需求不符合逻辑可以尊重用户需求
            d. 画风词:例如, cartoon, painting, painting style, etc，要和主体词相关性较强，例如iron man比较适合cartoon或movie
            e. 图片质量词:例如, high quality, masterpiece, etc.尽量避免低质量图片相关形容词
            f. 其他词:例如, happy, sad, etc.
            3. 对于你觉得对生成图像和用户意图相关性较强的短语，需要用英文括号圈起来，并在该短语后根据重要程度加上英文冒号和一个大于1不大于2的权重指数，例如(beautiful:1.6)；
            4. 你的翻译需要少用介词、助词、代词，多用实词。
            5. 短语之间不要有重复
            以下是一个示例：小女孩包饺子，翻译为：1 girl, (make dumpling:1.5), (Chinese New Year's Eve party:1.2), delicate appearance, vibrant red dress, high-quality ingredients, appealing posture, soft and fluffy texture, high quality, (masterpiece:1.6), realistic。
            以下是中文画图需求：{user_prompt}。
            只需给我最终答案，不需要解释
        """
    
    def _build_negative_prompt(self, negative_prompt):
        return f"""
            请把这段话翻译为英文,保留所有标点符号：{negative_prompt}。
            只需给我最终答案，不需要解释
        """

    def translate_img_generation_negative_prompt(self, negative_prompt):
        user_query = self._build_negative_prompt(negative_prompt)
        translated_prompt = self.model.chat(self.tokenizer, user_query, history=[], top_p=1,
                                                                        temperature=0.5)
        return translated_prompt[0]
    def translate_img_generation_prompt(self, user_prompt):
        user_query = self._build_prompt(user_prompt)
        translated_prompt = self.model.chat(self.tokenizer, user_query, history=[], top_p=1,
                                                                        temperature=0.5)
        return translated_prompt[0]

translator = ImageGenerationPromptTranslate() 

if __name__ == "__main__":
    translated_prompt = translator.translate_img_generation_prompt("女孩包饺子")
    print(translated_prompt)   
    while True:
        user_prompt = input("请输入中文画图需求：")
        if user_prompt == "exit":
            break
        translated_prompt = translator.translate_img_generation_prompt(user_prompt)
        print(translated_prompt)
    
