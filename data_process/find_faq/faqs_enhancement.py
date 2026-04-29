import os
import json
import time
import logging
import requests
from typing import List, Dict, Optional
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataAugmenter:
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com/v1/chat/completions",
                 model: str = "deepseek-chat", temperature: float = 0.7, max_tokens: int = 1024,
                 output_dir: str = "./augmented_data", retries: int = 3, delay: int = 2):
        """
        初始化数据增强器
        :param api_key: DeepSeek API密钥，默认从环境变量DEEPSEEK_API_KEY读取
        :param base_url: API端点URL
        :param model: 模型名称
        :param temperature: 生成温度
        :param max_tokens: 最大输出token数
        :param output_dir: 增强后数据输出目录
        :param retries: 重试次数
        :param delay: 重试间隔（秒）
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请提供DeepSeek API密钥，或设置环境变量DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_dir = output_dir
        self.retries = retries
        self.delay = delay
        os.makedirs(output_dir, exist_ok=True)

        # 定义增强方式列表（排除多轮对话）
        self.augmentation_methods = [
            "back_translation",      # 1. 回译增强
            "synonym_rewrite",       # 2. 同义改写
            "generalization",        # 3. 通用化改写
            "feature_derivation",    # 4. 特征衍生
            "sentence_transformation", # 5. 句式变换
            "perspective_shift",     # 6. 视角转换
            "scenario_expansion",    # 7. 情景扩展
            "professional_level",    # 9. 专业性分层（生成两个版本）
            "combine_taboos",        # 10. 结合其他相似禁忌问题
            "noise_injection",       # 11. 噪声注入
            "cross_culture",         # 12. 跨语言文化适配
            "analogy_transfer",      # 13. 类比迁移
            "negation_counterfactual", # 14. 否定/反事实生成
            "emotion_variation"      # 15. 情感与语气变换
        ]

        # 增强方式的中文名称（用于输出文件名）
        self.method_names = {
            "back_translation": "回译增强",
            "synonym_rewrite": "同义改写",
            "generalization": "通用化改写",
            "feature_derivation": "特征衍生",
            "sentence_transformation": "句式变换",
            "perspective_shift": "视角转换",
            "scenario_expansion": "情景扩展",
            "professional_level": "专业性分层",
            "combine_taboos": "结合禁忌",
            "noise_injection": "噪声注入",
            "cross_culture": "跨语言文化",
            "analogy_transfer": "类比迁移",
            "negation_counterfactual": "否定反事实",
            "emotion_variation": "情感变换"
        }

    def call_llm(self, prompt: str, expect_json: bool = True) -> Optional[str]:
        """调用DeepSeek API，返回响应文本"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        for attempt in range(self.retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                logging.warning(f"API调用失败 (尝试 {attempt+1}/{self.retries}): {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.delay)
                else:
                    logging.error(f"API调用最终失败: {e}")
                    return None
            except (KeyError, ValueError) as e:
                logging.error(f"响应解析失败: {e}")
                return None
        return None

    def parse_json_response(self, text: str) -> List[Dict[str, str]]:
        """解析LLM返回的JSON字符串，提取instruction和output列表"""
        try:
            # 清理可能的markdown代码块
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
            if isinstance(data, list):
                # 确保每个元素都有instruction和output
                valid = []
                for item in data:
                    if isinstance(item, dict) and "instruction" in item and "output" in item:
                        valid.append({"instruction": item["instruction"], "output": item["output"]})
                return valid
            elif isinstance(data, dict) and "instruction" in data and "output" in data:
                return [{"instruction": data["instruction"], "output": data["output"]}]
            else:
                logging.warning(f"返回的JSON格式不符合预期: {text[:100]}")
                return []
        except Exception as e:
            logging.error(f"JSON解析失败: {e}\n内容: {text[:200]}")
            return []

    # ---------- 增强方法 ----------
    def back_translation(self, entry: Dict) -> List[Dict]:
        """回译增强：将问答对翻译成英文再译回中文"""
        prompt = f"""请对以下中文问答对进行回译增强：先将整个问答对（包括问题和回答）翻译成英文，然后再将英文翻译回中文。注意检查答案的逻辑合理性。
        要求：最终结果必须是与原意一致但表述有所不同的中文问答对。
        请以JSON格式返回，格式为：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def synonym_rewrite(self, entry: Dict) -> List[Dict]:
        prompt = f"""请对以下问答对中的问题进行同义改写，生成一个不同表述但核心意图不变的新问题。
        回答内容可以保持不变，也可以稍作调整以适应新问题。
        请以JSON数组格式返回，数组中的每个元素包含"instruction"和"output"字段，例如：
        
            {{"instruction": "新问题1", "output": "对应回答1"}},

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def generalization(self, entry: Dict) -> List[Dict]:
        """通用化改写：将领域特定问答转化为通用场景下的问答"""
        prompt = f"""请将以下特定领域的问答对改写为通用场景下的问答，去除领域术语，保留核心知识，使其适用于更广泛的用户。
        以JSON格式返回单个问答对：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def feature_derivation(self, entry: Dict) -> List[Dict]:
        """特征衍生：如果原问答涉及年龄、性别、品种等特征，则按不同特征值生成新问答对"""
        # 首先判断是否包含特征关键词
        text = entry['instruction'] + entry['output']

        prompt = f"""原问答对涉及宠物特征（如年龄、性别、品种等）。请生成3-5个新的问答对，分别对应不同的特征组合（例如老年雄性金毛、中年雌性、幼年博美等，只是示例，不要局限在这几种组合），并相应调整问题与回答，保持核心知识不变。请注意不要有常识上的错误，例如"幼年犬生育"等组合
        以JSON数组格式返回，每个元素包含"instruction"和"output";如果不涉及任何宠物特征则返回空。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def sentence_transformation(self, entry: Dict) -> List[Dict]:
        """句式变换：将陈述句改为疑问句，或反之；否定句与肯定句转换"""
        prompt = f"""请对以下问答对进行句式变换，可以：
        - 将问题中的陈述句改为疑问句，或反之
        - 将回答中的否定句改为肯定句再转否定
        生成至少1个新问答对，保持核心含义不变。
        以JSON格式返回单个问答对：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def scenario_expansion(self, entry: Dict) -> List[Dict]:
        """情景扩展：添加具体场景细节，如时间、地点、紧急程度"""
        prompt = f"""请为以下问答对添加一个具体的场景细节（如深夜、急诊、乡村等，只是示例，不要局限在这几种组合），生成一个新问答对，保持核心知识不变。
        以JSON格式返回单个问答对：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def professional_level(self, entry: Dict) -> List[Dict]:
        """专业性分层：生成通俗版（对普通宠物主人）和专业版（对兽医或资深宠主）"""
        prompt = f"""请将以下问答对改写为两个版本：
        1. 通俗版：面向普通宠物主人，语言简单易懂，避免专业术语。
        2. 专业版：面向兽医或资深宠主，包含专业术语和详细解释。
        两个版本的问题可以相同或稍作调整。
        以JSON数组返回这两个新问答对。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def combine_taboos(self, entry: Dict) -> List[Dict]:
        """结合其他相似禁忌问题：生成复合问答"""
        prompt = f"""请结合其他类似的主题问答对，生成一个新的复合问答对，将原问答对与其他问答对结合讨论。
        以JSON格式返回单个问答对：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def noise_injection(self, entry: Dict) -> List[Dict]:
        """噪声注入：对问题加入错别字、口语词；回答插入冗余信息"""
        prompt = f"""请对以下问答对进行噪声注入：
        - 问题部分加入常见的错别字、口语词或网络用语（如“能网上买针给狗子自个儿打不？”，只是示例，不要局限在某几种噪声注入方式）。
        - 回答部分插入轻微冗余信息或同义重复，但保持关键信息完整。
        生成一个新问答对。
        以JSON格式返回单个问答对：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def cross_culture(self, entry: Dict) -> List[Dict]:
        """跨语言文化适配：将问题改为不同国家/地区的常见表达（如美式英语直译、日语式表达），再转回中文"""
        prompt = f"""请将以下问答对进行跨语言文化适配：
        模拟美式英语直译成中文的风格，或日本式中文表达（如“网购疫苗，自己打针，可以吗？”，只是示例，不要局限在这个主题与这种提问方式），生成一个具有异域文化风格的新问答对。
        以JSON格式返回单个问答对：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def analogy_transfer(self, entry: Dict) -> List[Dict]:
        """类比迁移：将“网购疫苗”类比为其他类似行为（如网购处方药、网购驱虫药）"""
        text = entry['instruction'] + entry['output']

        prompt = f"""请将原问答中的某些行为类比为其他类似的行为（如将网购疫苗擅自给宠物使用类比为网购处方药、网购驱虫药、网购注射器等，只是示例，不要局限于这种改变方式），生成一个新问答对，注意逻辑正确性。
        以JSON格式返回单个问答对：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def negation_counterfactual(self, entry: Dict) -> List[Dict]:
        """否定/反事实生成：构造假设性问题，测试模型对禁忌的坚持"""
        prompt = f"""请基于原问答构造一个假设性的反事实问题，(例如“如果我在网上买的疫苗是正规厂家，能自己打吗？”，只是示例，不要局限在这种假设方式)，并生成严谨的回答，强调禁忌不可违反。
        以JSON格式返回单个问答对：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    def emotion_variation(self, entry: Dict) -> List[Dict]:
        """情感与语气变换：将提问者的情绪改为焦虑、愤怒或恳求中的一种，生成一个新问答对，回答应保持专业但适当安抚情绪"""
        prompt = f"""请将原问答中的提问者情绪改为焦虑、愤怒或恳求中的一种，生成一个新问答对，回答应保持专业但适当安抚情绪。
        以JSON格式返回单个问答对：{{"instruction": "新问题", "output": "新回答"}}。

        原问答：
        问题：{entry['instruction']}
        回答：{entry['output']}
        """
        result = self.call_llm(prompt)
        if result:
            return self.parse_json_response(result)
        return []

    # ---------- 主流程 ----------
    def process_file(self, file_path: str):
        """处理单个JSON文件，为每种增强方式生成独立的输出文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]  # 确保是列表

        file_basename = os.path.splitext(os.path.basename(file_path))[0]

        # 对每种增强方式，收集所有条目生成的新样本
        for method in self.augmentation_methods:
            logging.info(f"正在处理文件 {file_basename}，增强方式：{method}")
            all_samples = []
            augment_func = getattr(self, method, None)
            if not augment_func:
                logging.warning(f"未找到方法 {method}，跳过")
                continue

            for entry in tqdm(data, desc=f"{method} 进度"):
                try:
                    samples = augment_func(entry)
                    if samples:
                        all_samples.extend(samples)
                except Exception as e:
                    logging.error(f"处理条目 {entry.get('metadata', {}).get('id', 'unknown')} 时出错: {e}")
                    continue

            # 如果生成了样本，写入文件
            if all_samples:
                output_filename = f"{file_basename}_{self.method_names.get(method, method)}.json"
                output_path = os.path.join(self.output_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_samples, f, ensure_ascii=False, indent=2)
                logging.info(f"已生成 {len(all_samples)} 个样本，保存至 {output_path}")
            else:
                logging.info(f"增强方式 {method} 未生成任何样本")

    def run(self, input_dir: str):
        """运行整个增强流程"""
        for filename in os.listdir(input_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(input_dir, filename)
                self.process_file(file_path)

if __name__ == "__main__":
    # 使用示例
    augmenter = DataAugmenter(
        api_key="sk-2eedab5b21954b6bb26f7461706642f2",
        output_dir="./augmented_output"
    )
    augmenter.run("faq_json/.")  # 输入文件夹路径