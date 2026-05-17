import json
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.core.config import BGE_MODEL_NAME, CHROMA_PERSIST_DIR
from src.json_loader import VetRAGDataLoader
from src.vector_store_chroma import ChromaVectorStore


EXPECTED_PHARMA_SCHEMA = "1.0"


def _validate_pharma_schema(file_path: str):
    """校验 pharmaceuticals.json 的 schema_version 与当前代码兼容。"""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("schema_version") if isinstance(data, dict) else None
    if version is None:
        print("警告：pharmaceuticals.json 缺少 schema_version，请尽快升级数据格式。")
    elif version != EXPECTED_PHARMA_SCHEMA:
        print(f"错误：pharmaceuticals.json schema_version={version}，期望={EXPECTED_PHARMA_SCHEMA}")
        print("请运行 scripts/enrich_pharmaceuticals.py 重新生成数据。")
        sys.exit(1)


def main():
    data_dir = os.path.join(project_root, "data")
    file_paths = [
        os.path.join(data_dir, "behaviors.json"),
        os.path.join(data_dir, "breeds.json"),
        os.path.join(data_dir, "cares.json"),
        os.path.join(data_dir, "diseases.json"),
        os.path.join(data_dir, "pharmaceuticals.json"),
        os.path.join(data_dir, "surgeries.json"),
    ]

    missing = [f for f in file_paths if not os.path.exists(f)]
    if missing:
        print("以下文件不存在:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    # 校验数据格式
    pharma_path = os.path.join(data_dir, "pharmaceuticals.json")
    if os.path.exists(pharma_path):
        _validate_pharma_schema(pharma_path)

    loader = VetRAGDataLoader()
    all_chunks = loader.load_all_files(file_paths)
    print(f"加载完成，共 {len(all_chunks)} 个文本块")

    vector_store = ChromaVectorStore(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="veterinary_rag",
        model_name=BGE_MODEL_NAME,
    )
    result = vector_store.add_chunks(all_chunks)
    print(f"向量化完成：新增 {result['added']}，跳过 {result['skipped']}，总计 {result['current_total']}")


if __name__ == "__main__":
    main()
