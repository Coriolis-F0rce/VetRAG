import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.json_loader import VetRAGDataLoader
from src.vector_store_chroma import ChromaVectorStore
from src.core.config import CHROMA_PERSIST_DIR, BGE_MODEL_NAME


def main():
    data_dir = os.path.join(current_dir, "data")
    file_paths = [
        os.path.join(data_dir, "behaviors.json"),
        os.path.join(data_dir, "breeds.json"),
        os.path.join(data_dir, "cares.json"),
        os.path.join(data_dir, "diseases.json"),
        os.path.join(data_dir, "surgeries.json"),
    ]

    missing = [f for f in file_paths if not os.path.exists(f)]
    if missing:
        print("以下文件不存在:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

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