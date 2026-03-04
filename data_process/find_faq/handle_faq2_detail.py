import json
import time
import requests
from typing import List, Dict, Tuple
import re


class DogMedicalFAQGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def parse_faq_content(self, file_content: str) -> Tuple[List[Dict], List[Dict]]:
        """解析FAQ文件内容，返回病例列表和追问链列表"""
        lines = file_content.strip().split('\n')

        cases = []
        chains = []
        current_system = ""

        # 解析状态
        parsing_cases = True
        parsing_chains = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否开始解析追问链部分
            if line.startswith('一、系统性诊断推理链'):
                parsing_cases = False
                parsing_chains = True
                continue

            # 解析病例部分
            if parsing_cases:
                # 检查是否是系统标题
                if line in ['消化系统', '呼吸系统', '皮肤系统', '泌尿系统', '神经系统',
                            '骨骼肌肉系统', '心血管系统', '内分泌系统', '血液与免疫系统',
                            '生殖系统', '眼耳系统', '中毒与急诊', '肿瘤科', '感染与寄生虫',
                            '幼犬与先天性疾病', '老年与退行性疾病', '行为与心理', '创伤与物理损伤']:
                    current_system = line
                    continue

                # 检查是否是病例行
                if ' - ' in line:
                    parts = line.split(' - ')
                    if len(parts) == 2:
                        symptom = parts[0].strip()
                        etiology = parts[1].strip()

                        # 提取病因列表
                        causes = []
                        # 处理多个病因的情况
                        if '或' in etiology:
                            etiology_parts = etiology.split('或')
                            for part in etiology_parts:
                                part = part.strip()
                                # 提取括号内的具体病因
                                match = re.search(r'\((.*?)\)', part)
                                if match:
                                    cause_type = part.split('(')[0].strip()
                                    specific_causes = match.group(1).split('、')
                                    causes.append({
                                        'type': cause_type,
                                        'specific': specific_causes
                                    })
                                else:
                                    causes.append({
                                        'type': part,
                                        'specific': []
                                    })
                        else:
                            # 单个病因
                            match = re.search(r'\((.*?)\)', etiology)
                            if match:
                                cause_type = etiology.split('(')[0].strip()
                                specific_causes = match.group(1).split('、')
                                causes.append({
                                    'type': cause_type,
                                    'specific': specific_causes
                                })
                            else:
                                causes.append({
                                    'type': etiology,
                                    'specific': []
                                })

                        cases.append({
                            'system': current_system,
                            'symptom': symptom,
                            'etiology': etiology,
                            'causes': causes,
                            'type': 'case'
                        })

            # 解析追问链部分
            elif parsing_chains:
                if '链' in line and '第' not in line:
                    # 新的追问链
                    current_chain = {
                        'name': line,
                        'rounds': []
                    }
                    chains.append(current_chain)
                elif '第' in line and '轮' in line:
                    # 追问链的轮次
                    round_info = {
                        'text': line,
                        'round_num': int(line[line.find('第') + 1:line.find('轮')])
                    }
                    if chains:
                        chains[-1]['rounds'].append(round_info)

        return cases, chains

    def generate_case_faq(self, case_data: Dict) -> Dict:
        """生成单个病例的FAQ"""
        system = case_data['system']
        symptom = case_data['symptom']
        etiology = case_data['etiology']

        # 构建主人视角的问题
        instruction = f"我的狗出现{symptom}，可能是什么问题？"

        # 构建prompt
        prompt = self._build_case_prompt(system, symptom, etiology)

        # 调用API
        response = self._call_api(prompt)

        # 解析响应
        try:
            result = json.loads(response)
            return {
                "instruction": instruction,
                "input": "",
                "output": result.get("output", ""),
                "metadata": {
                    "system": system,
                    "symptom": symptom,
                    "etiology": etiology,
                    "type": "single_case"
                }
            }
        except json.JSONDecodeError:
            # 如果API返回的不是JSON，使用原始响应作为output
            return {
                "instruction": instruction,
                "input": "",
                "output": response,
                "metadata": {
                    "system": system,
                    "symptom": symptom,
                    "etiology": etiology,
                    "type": "single_case"
                }
            }

    def generate_chain_faq(self, chain_data: Dict, chain_index: int) -> List[Dict]:
        """生成单个追问链的所有轮次FAQ"""
        chain_results = []
        chain_name = chain_data['name']
        rounds = chain_data['rounds']

        context = ""  # 用于存储上一轮的结论

        for round_data in rounds:
            round_text = round_data['text']
            round_num = round_data['round_num']

            # 构建prompt
            prompt = self._build_chain_prompt(chain_name, round_text, round_num, context, chain_index)

            # 调用API
            response = self._call_api(prompt)

            try:
                result = json.loads(response)
                output = result.get("output", "")

                # 构建instruction
                if round_num == 1:
                    instruction = f"{round_text.split('：')[1]}"
                else:
                    instruction = f"（{chain_name}-{round_num}）{round_text.split('：')[1]}"

                chain_results.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "metadata": {
                        "chain_name": chain_name,
                        "chain_index": chain_index,
                        "round": round_num,
                        "total_rounds": len(rounds),
                        "type": "chain_round"
                    }
                })

                # 更新上下文（取output的前100字）
                if output:
                    context = output[:100] + "..."

            except json.JSONDecodeError:
                # 使用原始响应
                chain_results.append({
                    "instruction": round_text,
                    "input": "",
                    "output": response,
                    "metadata": {
                        "chain_name": chain_name,
                        "chain_index": chain_index,
                        "round": round_num,
                        "total_rounds": len(rounds),
                        "type": "chain_round"
                    }
                })

            time.sleep(0.5)  # 避免API限制

        return chain_results

    def _build_case_prompt(self, system: str, symptom: str, etiology: str) -> str:
        """构建病例分析的prompt"""
        return f"""你是一名持有执业兽医资格证的小动物全科医学专家，具有15年临床经验。

请基于以下病例信息生成专业、准确、可操作的医学分析：

[DISEASE_SYSTEM] = {system}
[CLINICAL_SCENARIO] = {symptom}
[POSSIBLE_ETIOLOGIES] = {etiology}
[BREED_CONTEXT] = 无品种特异性
[AGE_CONTEXT] = 成年犬（需根据病因调整年龄考虑）

请严格按照以下结构组织回答：

临床优先度评估
- 紧急程度：[低/中/高/极高]
- 就医窗口：[家庭观察/24-48小时预约/12小时内急诊/立即急诊]

病理机制简述
用1-2句话解释疾病发生的生物学过程。

鉴别诊断思路
按可能性从高到低列出2-4个主要鉴别方向，每个方向包含具体疾病名称和关键区分特征。

就医前准备清单
- 携带物品
- 病史信息准备
- 家庭环境调整建议

药物治疗注意事项（如适用）
- 药物作用机制简述
- 标准给药方案
- 常见副作用
- 禁止联合使用的食物或药物（例如：头孢类药物期间禁止饮酒）

预后与长期管理
- 短期预后
- 长期预后
- 复发风险
- 定期监测建议

主人教育要点
1. 症状恶化的红色警报信号
2. 恢复期护理常见误区
3. 预防措施

免责声明
本建议基于有限的病例信息，不能替代执业兽医的面对面诊疗。

要求：
- 语言专业但易懂
- 基于循证医学原则
- 长度400-600字
- 使用中文，格式清晰但不要使用Markdown

请直接生成分析内容，不要包含额外解释。"""

    def _build_chain_prompt(self, chain_name: str, round_text: str, round_num: int,
                            context: str, chain_index: int) -> str:
        """构建追问链的prompt"""
        prompt = f"""你是一名资深兽医专家，正在处理一个犬类病例的追问链。

追问链主题：{chain_name}
当前轮次：第{round_num}轮
问题：{round_text}

"""

        if context:
            prompt += f"上一轮的关键结论：{context}\n\n"

        prompt += """请基于你的专业知识，提供深入、准确的分析。

你的回答应包含：
1. 核心答案：直接回答当前轮次的问题
2. 详细解释：提供医学原理和临床依据
3. 实用建议：给出可操作的建议
4. 注意事项：提示相关风险和禁忌

要求：
- 基于WSAVA/ACVIM临床共识
- 语言专业但清晰
- 300-500字
- 使用中文，格式清晰

请直接生成回答内容。"""

        return prompt

    def _call_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"API错误: {response.status_code}")
                return json.dumps({"error": f"API调用失败: {response.status_code}"})

        except Exception as e:
            print(f"API调用异常: {e}")
            return json.dumps({"error": f"连接异常: {str(e)}"})

    def batch_generate_all(self, file_content: str, output_file: str = "all_faqs.json"):
        """批量生成所有FAQ"""
        print("开始解析文件内容...")
        cases, chains = self.parse_faq_content(file_content)

        print(f"解析到 {len(cases)} 个病例")
        print(f"解析到 {len(chains)} 个追问链")

        all_results = []

        # 生成病例FAQ
        print("\n开始生成病例FAQ...")
        for i, case in enumerate(cases, 1):
            print(f"正在生成病例 {i}/{len(cases)}: {case['symptom']}")

            for attempt in range(3):
                try:
                    result = self.generate_case_faq(case)
                    if "error" not in result.get("output", ""):
                        all_results.append(result)
                        break
                    else:
                        print(f"  第{attempt + 1}次尝试失败")
                except Exception as e:
                    print(f"  生成失败: {e}")
                    time.sleep(2)

            time.sleep(1)  # 避免API限制

        # 生成追问链FAQ
        print("\n开始生成追问链FAQ...")
        for i, chain in enumerate(chains, 1):
            print(f"正在生成追问链 {i}/{len(chains)}: {chain['name']}")

            for attempt in range(3):
                try:
                    chain_results = self.generate_chain_faq(chain, i)
                    if chain_results:
                        all_results.extend(chain_results)
                        break
                    else:
                        print(f"  第{attempt + 1}次尝试失败")
                except Exception as e:
                    print(f"  生成失败: {e}")
                    time.sleep(2)

            time.sleep(1)

        # 保存结果
        print(f"\n保存结果到 {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"完成！共生成 {len(all_results)} 个FAQ条目")
        return all_results


def main():
    # 读取文件内容
    with open("faq2.txt", "r", encoding="utf-8") as f:
        file_content = f.read()

    # 初始化生成器
    # 注意：实际使用时请将API密钥存储在环境变量中
    api_key = "sk-2eedab5b21954b6bb26f7461706642f2"
    generator = DogMedicalFAQGenerator(api_key)

    # 批量生成所有FAQ
    results = generator.batch_generate_all(file_content, "faq2.json")

    # 输出统计信息
    case_count = sum(1 for r in results if r["metadata"]["type"] == "single_case")
    chain_count = sum(1 for r in results if r["metadata"]["type"] == "chain_round")

    print(f"\n生成统计:")
    print(f"- 病例FAQ: {case_count} 个")
    print(f"- 追问链条目: {chain_count} 个")
    print(f"- 总计: {len(results)} 个条目")


if __name__ == "__main__":
    main()