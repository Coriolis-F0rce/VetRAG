import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging


class DogCareDataDeduplicator:
    """犬只护理数据去重器"""

    def __init__(self, log_level=logging.INFO):
        """
        初始化去重器

        Args:
            log_level: 日志级别
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # 存储内容信息
        self.content_registry: Dict[str, Dict] = {}
        # 存储文件信息
        self.file_registry: Dict[str, Dict] = {}
        # 存储哈希值到内容的映射
        self.hash_to_content: Dict[str, str] = {}

    def calculate_content_hash(self, content: Dict) -> str:
        """
        计算内容的哈希值

        Args:
            content: 内容字典

        Returns:
            内容的MD5哈希值
        """
        # 转换为JSON字符串并标准化（排序键）
        content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        # 计算MD5哈希
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()

    def extract_content_id(self, data: Dict) -> Optional[str]:
        """
        从数据中提取内容ID

        Args:
            data: JSON数据

        Returns:
            内容ID或None
        """
        # 尝试从metadata中提取
        if "metadata" in data and "content_id" in data["metadata"]:
            return data["metadata"]["content_id"]

        # 尝试从文件结构推断（针对all_results_summary.json）
        if "results" in data:
            # 处理汇总文件，返回特殊标记
            return "all_summary"

        # 尝试从键名推断（针对batch文件）
        if "vaccine_schedule" in data:
            return "vaccine_schedule"
        elif "joint_care_guide" in data:
            return "joint_care_guide"
        elif "dog_regulations" in data:
            return "dog_regulations"
        elif "common_diseases" in data:
            return "common_diseases"
        elif "daily_care" in data:
            return "daily_care"

        return None

    def extract_content_type(self, data: Dict) -> str:
        """
        提取内容类型

        Args:
            data: JSON数据

        Returns:
            内容类型字符串
        """
        # 尝试直接获取content_type
        if "content_type" in data:
            return data["content_type"]

        # 尝试从metadata推断
        if "metadata" in data and "content_id" in data["metadata"]:
            content_id = data["metadata"]["content_id"]
            # 根据content_id映射到类型
            type_map = {
                "vaccine_schedule": "幼犬疫苗时间表",
                "joint_care_guide": "老年犬关节护理指南",
                "dog_regulations": "中国主要城市养犬管理规定",
                "common_diseases": "犬只常见疾病症状对照表",
                "daily_care": "犬只日常护理要点"
            }
            return type_map.get(content_id, "未知类型")

        return "未知类型"

    def extract_generation_time(self, data: Dict) -> Optional[datetime]:
        """
        提取生成时间

        Args:
            data: JSON数据

        Returns:
            datetime对象或None
        """
        # 尝试从metadata中提取
        if "metadata" in data and "generated_at" in data["metadata"]:
            try:
                return datetime.fromisoformat(data["metadata"]["generated_at"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass

        # 尝试从文件顶级提取
        if "completion_time" in data:
            try:
                return datetime.fromisoformat(data["completion_time"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass

        # 尝试从文件名提取（最后手段）
        return None

    def process_file(self, file_path: Path) -> Dict:
        """
        处理单个文件

        Args:
            file_path: 文件路径

        Returns:
            处理结果字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_info = {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                "data": data
            }

            self.file_registry[file_path.name] = file_info
            self.logger.info(f"已处理文件: {file_path.name}")

            return file_info

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误: {file_path.name} - {e}")
            return {}
        except Exception as e:
            self.logger.error(f"处理文件错误: {file_path.name} - {e}")
            return {}

    def extract_contents_from_data(self, data: Dict, source_file: str) -> List[Dict]:
        """
        从数据中提取所有内容块

        Args:
            data: JSON数据
            source_file: 源文件名

        Returns:
            内容块列表
        """
        contents = []

        # 情况1：汇总文件格式 (all_results_summary.json)
        if "results" in data:
            self.logger.info(f"检测到汇总文件格式: {source_file}")
            for content_id, content_data in data["results"].items():
                content_data["_source_file"] = source_file
                contents.append(content_data)

        # 情况2：批处理文件格式 (batch_x_results.json)
        elif any(key in data for key in ["vaccine_schedule", "joint_care_guide",
                                         "dog_regulations", "common_diseases", "daily_care"]):
            self.logger.info(f"检测到批处理文件格式: {source_file}")
            for content_data in data.values():
                if isinstance(content_data, dict):
                    content_data["_source_file"] = source_file
                    contents.append(content_data)

        # 情况3：单个内容文件格式
        elif "content_type" in data:
            self.logger.info(f"检测到单个内容文件格式: {source_file}")
            data["_source_file"] = source_file
            contents.append(data)

        return contents

    def register_content(self, content: Dict) -> Dict:
        """
        注册内容到去重系统

        Args:
            content: 内容字典

        Returns:
            注册结果字典
        """
        # 提取关键信息
        content_id = self.extract_content_id(content)
        content_type = self.extract_content_type(content)
        generation_time = self.extract_generation_time(content)

        if not content_id:
            self.logger.warning(f"无法提取内容ID: {content.get('content_type', '未知类型')}")
            return {"status": "error", "reason": "no_content_id"}

        # 计算内容哈希
        content_hash = self.calculate_content_hash(content)

        # 准备内容信息
        content_info = {
            "content_id": content_id,
            "content_type": content_type,
            "generation_time": generation_time,
            "content_hash": content_hash,
            "source_file": content.get("_source_file", "unknown"),
            "content": content
        }

        # 检查是否已存在
        if content_id in self.content_registry:
            existing = self.content_registry[content_id]

            # 情况1：内容完全相同（哈希相同）
            if existing["content_hash"] == content_hash:
                self.logger.info(f"发现完全相同的内容: {content_id} (源文件: {content['_source_file']})")
                return {"status": "duplicate", "identical": True}

            # 情况2：内容不同，选择更新的版本
            if generation_time and existing["generation_time"]:
                if generation_time > existing["generation_time"]:
                    self.logger.info(f"发现更新版本: {content_id} ({generation_time} > {existing['generation_time']})")
                    self.content_registry[content_id] = content_info
                    return {"status": "updated", "previous_time": existing["generation_time"]}
                else:
                    self.logger.info(f"发现更旧版本: {content_id} ({generation_time} <= {existing['generation_time']})")
                    return {"status": "older", "current_time": existing["generation_time"]}
            else:
                # 无法比较时间，保留第一个
                self.logger.warning(f"无法比较版本时间: {content_id}")
                return {"status": "conflict", "reason": "no_timestamp"}

        else:
            # 新内容
            self.content_registry[content_id] = content_info
            self.hash_to_content[content_hash] = content_id
            self.logger.info(f"注册新内容: {content_id} ({content_type})")
            return {"status": "new"}

    def process_directory(self, directory_path: str, pattern: str = "*.json") -> Dict:
        """
        处理目录中的所有JSON文件

        Args:
            directory_path: 目录路径
            pattern: 文件匹配模式

        Returns:
            处理统计信息
        """
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "total_contents": 0,
            "new_contents": 0,
            "duplicate_contents": 0,
            "updated_contents": 0,
            "older_contents": 0,
            "error_files": 0
        }

        directory = Path(directory_path)

        if not directory.exists():
            self.logger.error(f"目录不存在: {directory_path}")
            return stats

        # 查找所有JSON文件
        json_files = list(directory.glob(pattern))
        stats["total_files"] = len(json_files)

        self.logger.info(f"找到 {len(json_files)} 个JSON文件")

        for file_path in json_files:
            # 处理文件
            file_info = self.process_file(file_path)
            if not file_info:
                stats["error_files"] += 1
                continue

            stats["processed_files"] += 1

            # 提取内容
            contents = self.extract_contents_from_data(file_info["data"], file_path.name)

            for content in contents:
                stats["total_contents"] += 1
                result = self.register_content(content)

                if result["status"] == "new":
                    stats["new_contents"] += 1
                elif result["status"] == "duplicate":
                    stats["duplicate_contents"] += 1
                elif result["status"] == "updated":
                    stats["updated_contents"] += 1
                elif result["status"] == "older":
                    stats["older_contents"] += 1

        return stats

    def get_unique_contents(self) -> Dict:
        """
        获取去重后的所有内容

        Returns:
            去重后的内容字典
        """
        result = {}

        for content_id, info in self.content_registry.items():
            # 移除临时字段
            content = info["content"].copy()
            if "_source_file" in content:
                del content["_source_file"]

            result[content_id] = content

        return result

    def create_summary_report(self) -> Dict:
        """
        创建去重摘要报告

        Returns:
            摘要报告字典
        """
        summary = {
            "deduplication_summary": {
                "total_unique_contents": len(self.content_registry),
                "content_types": {},
                "latest_versions": []
            },
            "file_statistics": {
                "total_files_registered": len(self.file_registry),
                "files": list(self.file_registry.keys())
            }
        }

        # 按内容类型统计
        for info in self.content_registry.values():
            content_type = info["content_type"]
            if content_type not in summary["deduplication_summary"]["content_types"]:
                summary["deduplication_summary"]["content_types"][content_type] = 0
            summary["deduplication_summary"]["content_types"][content_type] += 1

            # 记录最新版本信息
            summary["deduplication_summary"]["latest_versions"].append({
                "content_id": info["content_id"],
                "content_type": info["content_type"],
                "generation_time": info["generation_time"].isoformat() if info["generation_time"] else None,
                "source_file": info["source_file"],
                "content_hash": info["content_hash"][:16] + "..."  # 只显示部分哈希
            })

        return summary

    def save_deduplicated_data(self, output_path: str, format_type: str = "summary") -> None:
        """
        保存去重后的数据

        Args:
            output_path: 输出路径
            format_type: 输出格式类型 ("summary" 或 "individual")
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if format_type == "summary":
            # 保存为all_results_summary格式
            unique_contents = self.get_unique_contents()

            summary_data = {
                "total_contents": len(unique_contents),
                "batches_processed": len(self.file_registry),
                "batch_size": 2,  # 根据实际情况调整
                "completion_time": datetime.now().isoformat(),
                "results": unique_contents,
                "deduplication_info": {
                    "generated_at": datetime.now().isoformat(),
                    "method": "content_hash_and_timestamp",
                    "total_files_processed": len(self.file_registry)
                }
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"已保存汇总文件: {output_path}")

        elif format_type == "individual":
            # 保存为单个文件
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            for content_id, info in self.content_registry.items():
                content = info["content"].copy()
                if "_source_file" in content:
                    del content["_source_file"]

                # 添加去重信息
                content["deduplication_info"] = {
                    "deduplicated_at": datetime.now().isoformat(),
                    "original_source": info["source_file"],
                    "generation_time": info["generation_time"].isoformat() if info["generation_time"] else None,
                    "content_hash": info["content_hash"]
                }

                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{content_id}_deduplicated_{timestamp}.json"
                filepath = output_dir / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)

                self.logger.info(f"已保存单个文件: {filename}")

    def find_duplicates(self) -> Dict[str, List[str]]:
        """
        查找所有重复项

        Returns:
            字典，键为内容哈希，值为文件列表
        """
        duplicates = {}

        # 按哈希值分组
        hash_groups = {}
        for file_name, file_info in self.file_registry.items():
            contents = self.extract_contents_from_data(file_info["data"], file_name)
            for content in contents:
                content_hash = self.calculate_content_hash(content)
                if content_hash not in hash_groups:
                    hash_groups[content_hash] = []
                hash_groups[content_hash].append(file_name)

        # 找出有重复的组
        for content_hash, files in hash_groups.items():
            if len(files) > 1:
                duplicates[content_hash] = files

        return duplicates

    def print_statistics(self) -> None:
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("犬只护理数据去重统计")
        print("=" * 50)

        print(f"\n文件统计:")
        print(f"  总文件数: {len(self.file_registry)}")
        for file_name in sorted(self.file_registry.keys()):
            print(f"  - {file_name}")

        print(f"\n内容统计:")
        print(f"  唯一内容数: {len(self.content_registry)}")

        # 按类型统计
        type_count = {}
        for info in self.content_registry.values():
            content_type = info["content_type"]
            type_count[content_type] = type_count.get(content_type, 0) + 1

        for content_type, count in sorted(type_count.items()):
            print(f"  - {content_type}: {count}")

        print(f"\n最新版本信息:")
        for content_id, info in self.content_registry.items():
            time_str = info["generation_time"].strftime("%Y-%m-%d %H:%M:%S") if info["generation_time"] else "未知"
            print(f"  - {content_id}: {time_str} (来自: {info['source_file']})")

        # 查找重复
        duplicates = self.find_duplicates()
        if duplicates:
            print(f"\n发现重复内容 ({len(duplicates)} 组):")
            for hash_val, files in duplicates.items():
                print(f"  哈希: {hash_val[:16]}...")
                for file in files:
                    print(f"    - {file}")

        print("\n" + "=" * 50)


# 使用示例
def main():
    """主函数示例"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建去重器
    deduplicator = DogCareDataDeduplicator(log_level=logging.INFO)

    # 假设文件在当前目录
    current_dir = Path(".")

    # 处理所有JSON文件
    stats = deduplicator.process_directory(current_dir)

    print("\n处理统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 打印详细统计
    deduplicator.print_statistics()

    # 创建并保存去重结果
    summary_report = deduplicator.create_summary_report()

    # 保存汇总文件
    deduplicator.save_deduplicated_data("deduplicated_all_results_summary.json", format_type="summary")

    # 保存单个文件
    deduplicator.save_deduplicated_data("deduplicated_individual_files", format_type="individual")

    # 保存摘要报告
    with open("deduplication_report.json", 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    print(f"\n去重完成！")
    print(f"  保存了汇总文件: deduplicated_all_results_summary.json")
    print(f"  保存了单个文件到目录: deduplicated_individual_files/")
    print(f"  保存了报告文件: deduplication_report.json")


# 快速使用函数
def quick_deduplicate(file_paths: List[str], output_file: str = "deduplicated_output.json"):
    """
    快速去重函数

    Args:
        file_paths: 文件路径列表
        output_file: 输出文件路径
    """
    deduplicator = DogCareDataDeduplicator()

    for file_path in file_paths:
        deduplicator.process_file(Path(file_path))

    # 提取并注册所有内容
    for file_name, file_info in deduplicator.file_registry.items():
        contents = deduplicator.extract_contents_from_data(file_info["data"], file_name)
        for content in contents:
            deduplicator.register_content(content)

    # 保存结果
    deduplicator.save_deduplicated_data(output_file, format_type="summary")

    print(f"去重完成！结果已保存到 {output_file}")
    print(f"唯一内容数: {len(deduplicator.content_registry)}")

    return deduplicator


if __name__ == "__main__":
    # 示例使用
    print("开始处理示例文件...")

    # 示例文件列表（根据实际文件名调整）
    example_files = [
        "all_results_summary.json",
        "batch_1_results.json",
        "batch_2_results.json",
        "batch_3_results.json",
        "daily_care_20260206_151716.json",
        "dog_regulations_20260206_151454.json",
        "joint_care_guide_20260206_151348.json",
        "vaccine_schedule_20260206_151320.json"
    ]

    # 检查文件是否存在
    available_files = []
    for file in example_files:
        if Path(file).exists():
            available_files.append(file)

    if available_files:
        deduplicator = quick_deduplicate(available_files, "cleaned_dog_care_data.json")
        deduplicator.print_statistics()
    else:
        print("未找到示例文件。请将文件放置在当前目录或指定正确的路径。")