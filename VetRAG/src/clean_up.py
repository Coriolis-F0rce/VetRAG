# cleanup_enhanced.py - 增强版清理脚本

import os
import shutil
import platform
from datetime import datetime


class Cleanup:
    """增强版清理工具"""

    def __init__(self):
        self.cleanup_log = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_action(self, action: str, status: str, details: str = ""):
        """记录清理动作"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "status": status,
            "details": details
        }
        self.cleanup_log.append(log_entry)
        print(f"{status} {action}: {details}")

    def cleanup_chroma_db(self, directories: list[str]):
        """清理ChromaDB目录"""
        for chroma_dir in directories:
            if os.path.exists(chroma_dir):
                try:
                    shutil.rmtree(chroma_dir)
                    self.log_action(
                        f"清理ChromaDB目录: {chroma_dir}",
                        "✓",
                        f"删除 {self._get_dir_size(chroma_dir)}"
                    )
                except Exception as e:
                    self.log_action(
                        f"清理ChromaDB目录: {chroma_dir}",
                        "✗",
                        str(e)
                    )
            else:
                self.log_action(
                    f"清理ChromaDB目录: {chroma_dir}",
                    "i",
                    "目录不存在"
                )

    def cleanup_cache(self, cache_patterns: list[str]):
        """清理缓存文件"""
        cleaned_count = 0
        error_count = 0

        for pattern in cache_patterns:
            if os.path.exists(pattern):
                try:
                    if os.path.isdir(pattern):
                        shutil.rmtree(pattern)
                        cleaned_count += 1
                        self.log_action(
                            f"清理目录: {pattern}",
                            "✓",
                            "目录已删除"
                        )
                    else:
                        os.remove(pattern)
                        cleaned_count += 1
                        self.log_action(
                            f"删除文件: {pattern}",
                            "✓",
                            "文件已删除"
                        )
                except Exception as e:
                    error_count += 1
                    self.log_action(
                        f"清理: {pattern}",
                        "✗",
                        str(e)
                    )

        self.log_action(
            "清理缓存文件",
            "✓",
            f"清理了 {cleaned_count} 个，失败 {error_count} 个"
        )

    def cleanup_huggingface_cache(self, specific_models: list[str] = None):
        """清理HuggingFace模型缓存"""
        system = platform.system()

        if system == "Windows":
            cache_base = os.path.join(os.environ.get("USERPROFILE", ""), ".cache", "huggingface")
        elif system in ["Linux", "Darwin"]:  # macOS
            cache_base = os.path.join(os.environ.get("HOME", ""), ".cache", "huggingface")
        else:
            self.log_action(
                "清理HuggingFace缓存",
                "✗",
                f"未知系统: {system}"
            )
            return

        if not os.path.exists(cache_base):
            self.log_action(
                "清理HuggingFace缓存",
                "i",
                "缓存目录不存在"
            )
            return

        if specific_models:
            cleaned = 0
            for model_name in specific_models:
                model_path = os.path.join(cache_base, "hub", model_name)
                if os.path.exists(model_path):
                    try:
                        shutil.rmtree(model_path)
                        cleaned += 1
                        self.log_action(
                            f"清理模型缓存: {model_name}",
                            "✓",
                            "模型缓存已删除"
                        )
                    except Exception as e:
                        self.log_action(
                            f"清理模型缓存: {model_name}",
                            "✗",
                            str(e)
                        )
            self.log_action(
                "清理HuggingFace模型缓存",
                "✓",
                f"清理了 {cleaned} 个模型"
            )
        else:
            # 清理整个huggingface缓存
            try:
                size_before = self._get_dir_size(cache_base)
                shutil.rmtree(cache_base)
                self.log_action(
                    "清理HuggingFace缓存",
                    "✓",
                    f"删除了 {size_before}"
                )
            except Exception as e:
                self.log_action(
                    "清理HuggingFace缓存",
                    "✗",
                    str(e)
                )

    def cleanup_incremental_state(self, state_dirs: list[str]):
        """清理增量状态目录"""
        for state_dir in state_dirs:
            if os.path.exists(state_dir):
                try:
                    # 备份状态文件
                    backup_dir = f"{state_dir}_backup_{self.timestamp}"
                    shutil.copytree(state_dir, backup_dir)

                    # 删除原目录
                    shutil.rmtree(state_dir)

                    self.log_action(
                        f"清理增量状态: {state_dir}",
                        "✓",
                        f"已备份到 {backup_dir}"
                    )
                except Exception as e:
                    self.log_action(
                        f"清理增量状态: {state_dir}",
                        "✗",
                        str(e)
                    )

    def cleanup_exports(self, export_dirs: list[str], keep_days: int = 7):
        """清理旧的导出文件"""
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        cleaned_files = 0

        for export_dir in export_dirs:
            if os.path.exists(export_dir):
                for item in os.listdir(export_dir):
                    item_path = os.path.join(export_dir, item)

                    try:
                        mtime = os.path.getmtime(item_path)
                        if mtime < cutoff_time:
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                            cleaned_files += 1
                    except:
                        continue

        self.log_action(
            "清理旧导出文件",
            "✓",
            f"清理了 {cleaned_files} 个文件/目录 (超过 {keep_days} 天)"
        )

    def cleanup_chunks_cache(self, cache_files: list[str]):
        """清理chunks缓存文件"""
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                try:
                    # 备份缓存
                    backup_file = f"{cache_file}.backup_{self.timestamp}"
                    shutil.copy2(cache_file, backup_file)

                    # 删除原文件
                    os.remove(cache_file)

                    self.log_action(
                        f"清理缓存文件: {cache_file}",
                        "✓",
                        f"已备份到 {backup_file}"
                    )
                except Exception as e:
                    self.log_action(
                        f"清理缓存文件: {cache_file}",
                        "✗",
                        str(e)
                    )

    def _get_dir_size(self, path: str) -> str:
        """获取目录大小"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)

        # 转换为易读格式
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0

        return f"{total_size:.1f} TB"

    def save_cleanup_log(self, log_file: str = "cleanup_log.json"):
        """保存清理日志"""
        import json
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": self.timestamp,
                "actions": self.cleanup_log,
                "summary": {
                    "total_actions": len(self.cleanup_log),
                    "success_count": sum(1 for log in self.cleanup_log if log['status'] == '✓'),
                    "error_count": sum(1 for log in self.cleanup_log if log['status'] == '✗')
                }
            }, f, ensure_ascii=False, indent=2)

        self.log_action(
            "保存清理日志",
            "✓",
            f"日志已保存到 {log_file}"
        )

    def run_full_cleanup(self):
        """运行完整清理流程"""
        print("=" * 60)
        print("开始完整清理流程")
        print("=" * 60)

        # 1. 清理ChromaDB
        print("\n1. 清理ChromaDB向量数据库")
        self.cleanup_chroma_db([
            "../chroma_db",
            "./chroma_db",
            "./chroma_db_bge",
            "../chroma_db_bge"
        ])

        # 2. 清理Python缓存
        print("\n2. 清理Python缓存")
        self.cleanup_cache([
            "../__pycache__",
            "./__pycache__",
            "../.chroma_cache",
            "./.chroma_cache",
            "*.pyc",
            "*.pyo"
        ])

        # 3. 清理HuggingFace缓存
        print("\n3. 清理HuggingFace模型缓存")
        self.cleanup_huggingface_cache([
            "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
            "models--BAAI--bge-large-zh-v1.5"
        ])

        # 4. 清理增量状态
        print("\n4. 清理增量状态目录")
        self.cleanup_incremental_state([
            "./incremental_state",
            "../incremental_state"
        ])

        # 5. 清理缓存文件
        print("\n5. 清理缓存文件")
        self.cleanup_chunks_cache([
            "./chunks_cache.pkl",
            "../chunks_cache.pkl",
            "./raw_chunks.pkl"
        ])

        # 6. 清理旧导出
        print("\n6. 清理旧导出文件")
        self.cleanup_exports([
            "./exports",
            "../exports"
        ], keep_days=7)

        # 7. 保存日志
        print("\n7. 保存清理日志")
        self.save_cleanup_log()

        print("\n" + "=" * 60)
        print("清理完成！")
        print("=" * 60)

        # 显示摘要
        success_count = sum(1 for log in self.cleanup_log if log['status'] == '✓')
        total_count = len(self.cleanup_log)
        print(f"执行了 {total_count} 个清理操作，成功 {success_count} 个")


def main():
    """主清理函数"""
    cleanup = Cleanup()
    cleanup.run_full_cleanup()


if __name__ == "__main__":
    main()