import os
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

import dotenv
dotenv.load_dotenv()

# --- 2. آماده‌سازی فایل و دایرکتوری‌ها ---
DATA_DIR = "data4"
STORAGE_DIR = "storage"
FILE_PATH = os.path.join(DATA_DIR, "iran_history.txt")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# --- 3. تنظیمات مدل و شمارش توکن ---
# تنظیم مدل زبان و شمارنده توکن برای محاسبه هزینه‌ها
Settings.llm = OpenAI(model="gpt-4.1-nano", temperature=0.8)
token_counter = TokenCountingHandler()
Settings.callback_manager = CallbackManager([token_counter])


try:
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
    print("✅ ایندکس از حافظه محلی بارگذاری شد.")
except:
    print(" ایندکس موجود یافت نشد. در حال ساخت ایندکس جدید...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"📄 فایل '{FILE_PATH}' با موفقیت بارگذاری شد.")


    index = VectorStoreIndex.from_documents(documents)
    print("✅ ایندکس‌گذاری با موفقیت انجام شد.")

    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"💾 ایندکس در دایرکتوری '{STORAGE_DIR}' ذخیره شد.")



base_query_engine = index.as_query_engine(similarity_top_k=3)

# سپس ابزار جستجو را تعریف می‌کنیم
query_engine_tools = [
    QueryEngineTool(
        query_engine=base_query_engine,
        metadata=ToolMetadata(
            name="stroe labtop",
            description="شما یک فروشنده حرفه و خبره لب تاپ هستید",
        ),
    ),
]

# 4. ساخت موتور جستجوی اصلی با قابلیت تولید زیر-سوال
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=False, # برای سادگی در این مثال از حالت همزمان استفاده می‌کنیم
)


complex_query = "مقایسه دو لب تاپ hp envy 15  با lenevo thinkpad p1"
print(f"\n❓ پرسش اصلی: {complex_query}\n")

# ریست کردن شمارنده توکن قبل از اجرای پرسش
token_counter.reset_counts()

# اجرای پرسش
response = query_engine.query(complex_query)

# --- 7. نمایش خروجی‌ها ---
print("✔️ --- پاسخ نهایی ---")
print(str(response))
print("=" * 25)

# 5. نمایش زیر-سوالات تولید شده
generated_questions = []
for source_node in response.source_nodes:
    # زیرسوالات در متادیتای نودهای منبع ذخیره می‌شوند
    if "sub_question" in source_node.metadata:
        generated_questions.append(source_node.metadata["sub_question"])

if generated_questions:
    print("\n🔍 --- زیر-سوالات تولید شده برای جستجوی بهتر ---")
    # استفاده از set برای نمایش سوالات منحصربه‌فرد
    for i, q in enumerate(set(generated_questions), 1):
        print(f"  {i}: {q.strip()}")
    print("-" * 25)
else:
    print("\nℹ️ هیچ زیر-سوالی تولید نشد (احتمالاً سوال اصلی به اندازه کافی ساده بوده است).")


embedding_tokens = token_counter.total_embedding_token_count
llm_tokens = token_counter.llm_token_counts

print("\n📊 --- آمار توکن‌های استفاده شده ---")
print(f"  توکن‌های Embedding (برای ایندکس‌گذاری): {embedding_tokens}")
print(f"  توکن‌های LLM (پرسش و پاسخ): {llm_tokens}")
print("-" * 25)


