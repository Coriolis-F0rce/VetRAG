from modelscope import snapshot_download

model_dir = snapshot_download(
    'qwen/Qwen3-0.6B',
    cache_dir='D:\Backup\PythonProject2\VetRAG\models\Qwen3-0.6B',
    revision='master'
)
print(f"✅ 下载完成！模型保存至: {model_dir}")