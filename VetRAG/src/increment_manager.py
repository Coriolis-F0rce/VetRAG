# incremental_manager.py - 增量更新和状态管理

import os
import json
import hashlib
import pickle
from datetime import datetime
from typing import List, Dict, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class FileStatus:
    """文件状态"""
    file_path: str
    file_hash: str
    file_size: int
    modified_time: float
    processed_time: str
    chunks_count: int
    added_count: int
    skipped_count: int
    content_types: List[str]
    success: bool
    error: str = ""


class IncrementalManager:
    """增量更新管理器"""

    def __init__(self, state_dir="./incremental_state"):
        self.state_dir = state_dir
        self.state_file = os.path.join(state_dir, "state.json")
        self.file_status_dir = os.path.join(state_dir, "file_status")

        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(self.file_status_dir, exist_ok=True)

        # 加载状态
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """加载系统状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        # 初始状态
        return {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_files": 0,
            "total_chunks": 0,
            "content_type_stats": {},
            "file_paths": []
        }

    def _save_state(self):
        """保存系统状态"""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def get_file_status(self, file_path: str) -> FileStatus:
        """获取文件状态"""
        status_file = os.path.join(self.file_status_dir, f"{hashlib.md5(file_path.encode()).hexdigest()}.json")

        if os.path.exists(status_file):
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return FileStatus(**data)
            except:
                pass

        # 创建新状态
        file_stats = os.stat(file_path)
        return FileStatus(
            file_path=file_path,
            file_hash=self.calculate_file_hash(file_path),
            file_size=file_stats.st_size,
            modified_time=file_stats.st_mtime,
            processed_time="",
            chunks_count=0,
            added_count=0,
            skipped_count=0,
            content_types=[],
            success=False
        )

    def save_file_status(self, status: FileStatus):
        """保存文件状态"""
        status_file = os.path.join(
            self.file_status_dir,
            f"{hashlib.md5(status.file_path.encode()).hexdigest()}.json"
        )

        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(status), f, ensure_ascii=False, indent=2)

        # 更新全局状态
        if status.success:
            if status.file_path not in self.state["file_paths"]:
                self.state["file_paths"].append(status.file_path)
                self.state["total_files"] += 1

            self.state["total_chunks"] += status.added_count

            # 更新内容类型统计
            for content_type in status.content_types:
                if content_type not in self.state["content_type_stats"]:
                    self.state["content_type_stats"][content_type] = 0
                self.state["content_type_stats"][content_type] += 1

        self._save_state()

    def is_file_modified(self, file_path: str) -> bool:
        """检查文件是否已修改"""
        if not os.path.exists(file_path):
            return True

        status = self.get_file_status(file_path)

        # 如果从未处理过
        if not status.processed_time:
            return True

        # 检查文件哈希
        current_hash = self.calculate_file_hash(file_path)
        if current_hash != status.file_hash:
            return True

        # 检查修改时间
        current_mtime = os.stat(file_path).st_mtime
        if current_mtime > status.modified_time:
            return True

        return False

    def get_new_or_modified_files(self, file_paths: List[str]) -> List[str]:
        """获取新的或修改过的文件"""
        return [
            file_path for file_path in file_paths
            if self.is_file_modified(file_path)
        ]

    def get_system_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            "system_info": {
                "version": self.state.get("version", "2.0"),
                "created_at": self.state.get("created_at"),
                "last_updated": self.state.get("last_updated")
            },
            "file_stats": {
                "total_files": self.state.get("total_files", 0),
                "total_chunks": self.state.get("total_chunks", 0),
                "processed_files": len(self.state.get("file_paths", []))
            },
            "content_type_distribution": self.state.get("content_type_stats", {}),
            "recent_files": self.state.get("file_paths", [])[-10:]  # 最近10个文件
        }

    def cleanup_old_status(self, days_old: int = 30):
        """清理旧的状态文件"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        removed_count = 0

        for status_file in Path(self.file_status_dir).glob("*.json"):
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                processed_time = data.get("processed_time", "")
                if processed_time:
                    processed_dt = datetime.fromisoformat(processed_time.replace('Z', '+00:00'))
                    if processed_dt.timestamp() < cutoff_time:
                        os.remove(status_file)
                        removed_count += 1
            except:
                continue

        print(f"清理了 {removed_count} 个旧状态文件")
        return removed_count