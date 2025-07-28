import os
import dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

# --- 1. بارگذاری متغیرهای محیطی از .env
dotenv.load_dotenv()

# --- 2. تنظیمات FastAPI
app = FastAPI()

# --- 3. تنظیمات CORS برای اتصال از فرانت
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در تولید بهتره محدود بشه
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. مدل ورودی سوال
class QuestionRequest(BaseModel):
    question: str

# --- 5. تنظیمات مسیرها و مدل LLM
DATA_DIR = "data4"
STORAGE_DIR = "storage"
FILE_PATH = os.path.join(DATA_DIR, "laptop.txt")

# تنظیم مدل LLM (مدل مناسب را انتخاب کن)
Settings.llm = OpenAI(model="gpt-4.1-nano", temperature=0.8)

token_counter = TokenCountingHandler()
Settings.callback_manager = CallbackManager([token_counter])

# --- 6. ساخت یا بارگذاری ایندکس
try:
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
    print("✅ ایندکس با موفقیت از حافظه بارگذاری شد.")
except Exception as e:
    print("❌ ایندکس پیدا نشد. در حال ساخت ایندکس جدید...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print("✅ ایندکس جدید ساخته و ذخیره شد.")

# --- 7. تنظیم Sub-Question Query Engine
base_query_engine = index.as_query_engine(similarity_top_k=3)
query_engine_tools = [
    QueryEngineTool(
        query_engine=base_query_engine,
        metadata=ToolMetadata(
            name="store_laptop",
            description="شما یک فروشنده باتجربه لپ‌تاپ هستید و به مشتریان مشاوره می‌دهید.",
        ),
    )
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=False,
)

# --- 8. مسیر API اصلی برای پاسخ به سوالات
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    token_counter.reset_counts()

    response = query_engine.query(question)

    # استخراج زیرسوالات اگر وجود داشته باشد
    generated_questions = []
    for node in response.source_nodes:
        if "sub_question" in node.metadata:
            generated_questions.append(node.metadata["sub_question"])

    return {
        "answer": str(response),
        "sub_questions": list(set(q.strip() for q in generated_questions if q.strip())),
        "token_stats": {
            "embedding_tokens": token_counter.total_embedding_token_count,
            "llm_tokens": token_counter.llm_token_counts,
        },
    }
