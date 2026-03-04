import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.rag_pipeline import VetRAGPipeline

def main():
    data_dir = "data"
    vector_store_type = "chroma"
    file_paths = [
        os.path.join(data_dir, "behaviors.json"),
        os.path.join(data_dir, "breeds.json"),
        os.path.join(data_dir, "cares.json"),
        os.path.join(data_dir, "diseases.json"),
        os.path.join(data_dir, "surgeries.json")
    ]

    missing_files = [f for f in file_paths if not os.path.exists(f)]
    if missing_files:
        print("以下文件不存在:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    pipeline = VetRAGPipeline(vector_store_type=vector_store_type)

    chunks = pipeline.build_from_files(file_paths)

    pipeline.save("veterinary_rag_system")
    print(f"\n构建完成！总共处理了 {len(chunks)} 个文本块")
    print("向量索引已保存到: veterinary_rag_system")

if __name__ == "__main__":
    main()