import json
import requests
import time


def call_deepseek_api(api_key: str = None, prompts_file: str = "find_behaviors/api_prompts.json",
                      output_file: str = "find_behaviors/api_responses.json"):
    """调用DeepSeek API处理prompts"""

    if not api_key:
        api_key = "sk-2eedab5b21954b6bb26f7461706642f2"

    try:
        print(f"加载prompts文件: {prompts_file}")
        with open(prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'prompts' not in data:
            print("错误: prompts文件格式不正确")
            return

        prompts = data['prompts']
        print(f"找到 {len(prompts)} 个prompts")

        base_url = "https://api.deepseek.com"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        results = []
        success_count = 0
        fail_count = 0

        print("开始处理...")
        for i, prompt_data in enumerate(prompts):
            if (i + 1) % 10 == 0 or i == 0 or i == len(prompts) - 1:
                print(f"进度: {i + 1}/{len(prompts)}")

            try:
                api_request = prompt_data.get('api_request')
                if not api_request:
                    api_request = {
                        "model": "deepseek-chat",
                        "messages": [
                            {
                                "role": "system",
                                "content": "你是一位犬行为学专家，请严格按照要求输出JSON格式，不添加任何额外文本。"
                            },
                            {
                                "role": "user",
                                "content": prompt_data['prompt']
                            }
                        ],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"}
                    }

                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=api_request,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        'id': prompt_data['id'],
                        'original_name': prompt_data['original_name'],
                        'response': result
                    })
                    success_count += 1
                else:
                    results.append({
                        'id': prompt_data['id'],
                        'original_name': prompt_data['original_name'],
                        'error': f"HTTP {response.status_code}: {response.text[:100]}"
                    })
                    fail_count += 1

            except requests.exceptions.RequestException as e:
                results.append({
                    'id': prompt_data['id'],
                    'original_name': prompt_data['original_name'],
                    'error': str(e)
                })
                fail_count += 1

            time.sleep(1)

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_processed': len(results),
                'successful': success_count,
                'failed': fail_count,
                'results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n处理完成！")
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        print(f"结果已保存到: {output_file}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {prompts_file}")
    except Exception as e:
        print(f"处理发生错误: {e}")


def create_api_prompts_from_txt(input_file: str = "find_behaviors/dog_behaviors.txt", output_file: str = "find_behaviors/api_prompts.json"):
    """从清洗后的txt文件创建API prompts"""

    print(f"从 {input_file} 创建API prompts...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        prompts = []
        entry_count = 0

        json_template = {
             "behavior": {
                "name": "",
                "category": "",
                "description": "",
                "antecedents": "",
                "consequences": "",
                "meaning":"",
                "evaluation":"",
                "function": "",
                "intervention_level": "",
                "resource":""
                }
            }

        for line in lines:
            line = line.strip()
            if not line or '提取到的行为条目' in line or '==' in line:
                continue

            # 解析行为条目
            if line[0].isdigit() and '. ' in line:
                entry_count += 1

                # 构建prompt
                prompt = f"""你是一位犬行为学专家。请将以下行为描述标准化为JSON格式。

原始数据：
{line}

要求：
1. 行为名称：使用专业术语
2. 分类：必选其一 → 社交沟通/生物本能/情绪表达/习得行为
3. 描述：客观、可观察
4. 场景：具体触发条件
5. 含义：推测需谨慎，标明"可能"而非"一定"
6. 评估：基于科学共识判断正常范围
7. 干预等级：必选其一 → 无/观察、低/自我调节、中/需要管理、高/立即干预、紧急/需专业行为兽医
8. 知识库来源：列出该行为解释所依据的主要理论或研究者

关键点：
- 信息不足时留空[]或""
- 仅输出JSON，不加额外文本

输出格式：
{json.dumps(json_template, indent=2, ensure_ascii=False)}"""

                # 构建API请求
                api_request = {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一位犬行为学专家，请严格按照要求输出JSON格式，不添加任何额外文本。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }

                prompts.append({
                    'id': str(entry_count),
                    'original_name': line.split('. ')[1].split(':')[0] if ': ' in line else line,
                    'prompt': prompt,
                    'api_request': api_request
                })

        # 保存prompts
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_prompts': len(prompts),
                'prompts': prompts
            }, f, indent=2, ensure_ascii=False)

        print(f"创建了 {len(prompts)} 个API prompts")
        print(f"已保存到: {output_file}")

        return prompts

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"创建prompts发生错误: {e}")


def main():
    """主函数"""

    print("=" * 40)
    print("犬行为数据API处理流水线")
    print("=" * 40)

    # 步骤1: 从txt文件创建API prompts
    print("\n步骤1: 创建API prompts...")
    prompts = create_api_prompts_from_txt("find_behaviors/dog_behaviors.txt")

    if not prompts:
        print("无法继续，请检查输入文件。")
        return

    # 步骤2: 调用DeepSeek API
    print("\n步骤2: 调用DeepSeek API...")

    # 安全提示
    print("注意: 将使用默认API密钥，如需更改请修改脚本。")
    print("开始调用API，这可能需要几分钟...")

    call_deepseek_api()

    print("\n" + "=" * 40)
    print("流水线处理完成！")
    print("=" * 40)


if __name__ == "__main__":
    main()