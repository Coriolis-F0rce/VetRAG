"""
Qwen3-1.7B 模型下载脚本 — ModelScope 源
适用平台：AutoDL / 本地
"""
from modelscope import snapshot_download

model_name = "Qwen/Qwen3-1.7B"
cache_dir  = "/root/autodl-tmp/huggingface/models/Qwen3-1.7B"

model_dir = snapshot_download(
    model_name,
    cache_dir=cache_dir,
    revision="master",
)
print(f"Qwen3-1.7B 下载完成 → {model_dir}")
