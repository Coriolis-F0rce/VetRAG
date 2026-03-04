import os
import sys
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from rag_interface import RAGInterface

# ========== 配置 ==========
CHROMA_DIR = os.path.join(current_dir, "chroma_db")
MODEL_PATH = r"D:\Backup\PythonProject2\VetRAG\models_finetuned\qwen3-finetuned"
# =========================

rag = RAGInterface(
    chroma_persist_dir=CHROMA_DIR,
    generator_model_path=MODEL_PATH
)

app = FastAPI(title="兽医RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录，使 index.html 可通过 / 访问
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # 直接返回 static/index.html 的内容
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

@app.get("/stats")
async def get_stats():
    return rag.get_stats()

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    question = data.get("question", "")
    top_k = data.get("top_k", 5)
    if not question:
        return {"error": "问题不能为空"}
    result = rag.query(question, top_k=top_k)
    return result

@app.get("/stream")
async def stream(question: str, top_k: int = 3, threshold: float = 0.5):
    if not question:
        return StreamingResponse(
            iter(["data: " + json.dumps({"error": "问题不能为空"}) + "\n\n"]),
            media_type="text/event-stream"
        )

    search_results = rag.vector_store.search(question, n_results=top_k)
    all_retrieved = search_results.get("results", [])
    valid_docs = [doc for doc in all_retrieved if doc.get("similarity", 0) >= threshold]

    context_parts = []
    for doc in valid_docs:
        content = rag._clean_document(doc["document"])
        if len(content) > 500:
            content = content[:500] + "…"
        context_parts.append(f"[相关文档] {content}")
    context = "\n\n".join(context_parts)

    system_msg = (
        "你是一个专业的兽医助手，同时也需要以温暖、共情的态度回答宠物主人的情感困惑。\n"
        "要求：\n"
        "1. 回答应简洁、清晰，直接针对问题，不要添加无关信息。\n"
        "2. 不要输出参考资料中的原始格式（如 JSON、代码块、Markdown 表格）。\n"
        "3. 不要添加免责声明、来源说明或注释。\n"
        "4. 回答应使用自然、流畅的段落，每段不超过 3 句话。\n"
        "5. 如果参考资料与问题不甚相关，请从你的语料库中进行适当分析，不要自行编造。"
    )
    prompt = rag.generator.build_chat_prompt(
        system=system_msg,
        user=question,
        context=context
    )

    async def event_generator():
        try:
            for token in rag.generator.async_stream_generate(prompt):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)