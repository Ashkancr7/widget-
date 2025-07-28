import os
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox

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

# --- 1. تنظیمات اولیه و کلید API ---
# اطمینان حاصل کنید که کلید OpenAI API شما در متغیرهای محیطی سیستم تنظیم شده است
if not os.getenv("OPENAI_API_KEY"):
    # چون هنوز پنجره اصلی ساخته نشده، از یک پاپ‌آپ ساده Tkinter استفاده می‌کنیم
    root = tk.Tk()
    root.withdraw() # مخفی کردن پنجره اصلی
    messagebox.showerror(
        "خطا: کلید API یافت نشد",
        "لطفاً قبل از اجرای برنامه، متغیر محیطی OPENAI_API_KEY را تنظیم کنید.",
    )
    sys.exit()

# --- 2. آماده‌سازی دایرکتوری‌ها و فایل نمونه ---
DATA_DIR = "data"
STORAGE_DIR = "storage"

def setup_directories():
    """ایجاد دایرکتوری‌های مورد نیاز و فایل داده نمونه."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        sample_file_path = os.path.join(DATA_DIR, "medical_equipment.txt")
        with open(sample_file_path, "w", encoding="utf-8") as f:
            f.write(
                "دستگاه الکتروکاردیوگرام (ECG) برای ثبت فعالیت الکتریکی قلب استفاده می‌شود. "
                "این دستگاه سیگنال‌های الکتریکی را از طریق الکترودهایی که به پوست بیمار متصل می‌شوند، دریافت می‌کند. "
                "دستگاه MRI یا تصویربرداری تشدید مغناطیسی، از میدان‌های مغناطیسی قوی و امواج رادیویی برای ایجاد تصاویر دقیق از اندام‌ها و بافت‌های بدن استفاده می‌کند. "
                "این روش برای تشخیص تومورها، آسیب‌های بافتی و مشکلات مفصلی بسیار کاربردی است. "
                "دستگاه ونتیلاتور یا تنفس مصنوعی، به بیمارانی که قادر به تنفس خود به خودی نیستند، کمک می‌کند. "
                "این دستگاه هوا را با فشار مثبت به داخل ریه‌ها می‌فرستد. "
                "پمپ انفوزیون برای تزریق دقیق و کنترل‌شده مایعات، داروها یا مواد مغذی به سیستم گردش خون بیمار به کار می‌رود."
            )

    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

def initialize_llama_index():
    """
    بارگذاری یا ساخت ایندکس و موتور جستجو.
    این تابع ممکن است زمان‌بر باشد.
    """
    # --- تنظیمات مدل و شمارش توکن ---
    Settings.llm = OpenAI(model="gpt-4", temperature=0.7)
    token_counter = TokenCountingHandler()
    Settings.callback_manager = CallbackManager([token_counter])

    # --- بارگذاری یا ساخت ایندکس ---
    try:
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        initial_status = "✅ ایندکس با موفقیت بارگذاری شد."
    except Exception:
        initial_status = "⏳ ایندکس یافت نشد. در حال ساخت ایندکس جدید..."
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        initial_status = "✅ ایندکس جدید ساخته و ذخیره شد."

    # --- ساخت موتور جستجو ---
    base_query_engine = index.as_query_engine(similarity_top_k=3)
    query_engine_tools = [
        QueryEngineTool(
            query_engine=base_query_engine,
            metadata=ToolMetadata(
                name="تجهیزات_پزشکی",
                description="ارائه اطلاعات در مورد تجهیزات پزشکی و نحوه کار آن‌ها",
            ),
        ),
    ]
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools, use_async=False
    )
    
    return query_engine, token_counter, initial_status

class LlamaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LlamaIndex GUI with Tkinter")
        self.root.geometry("700x700")

        self.query_engine = None
        self.token_counter = None

        # --- ساخت ویجت‌ها ---
        self.title_label = tk.Label(root, text="برنامه پرسش و پاسخ با LlamaIndex", font=("Tahoma", 16, "bold"))
        self.title_label.pack(pady=10)

        self.status_label = tk.Label(root, text="...در حال راه‌اندازی اولیه", relief="sunken", anchor="w", padx=5)
        self.status_label.pack(fill="x", padx=10, pady=5)

        # فریم برای ورودی کاربر
        input_frame = tk.Frame(root)
        input_frame.pack(fill="x", padx=10, pady=5)
        self.input_label = tk.Label(input_frame, text=":سوال خود را اینجا وارد کنید")
        self.input_label.pack(side="right")
        self.input_entry = tk.Entry(input_frame, font=("Tahoma", 12), justify='right')
        self.input_entry.pack(fill="x", expand=True, side="left")

        self.ask_button = tk.Button(root, text="بپرس", command=self.start_query_thread, state="disabled", bg="green", fg="white")
        self.ask_button.pack(pady=5)

        # فریم برای خروجی‌ها
        self.output_label = tk.Label(root, text=":پاسخ نهایی")
        self.output_label.pack(anchor="e", padx=10)
        self.output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15, font=("Tahoma", 11))
        self.output_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_text.config(state="disabled")

        self.subq_label = tk.Label(root, text=":زیرسوال‌های تولید شده")
        self.subq_label.pack(anchor="e", padx=10)
        self.subq_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=5, font=("Tahoma", 10))
        self.subq_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.subq_text.config(state="disabled")

        self.tokens_label = tk.Label(root, text=":آمار توکن‌ها")
        self.tokens_label.pack(anchor="e", padx=10)
        self.tokens_info_label = tk.Label(root, text="", relief="sunken", anchor="w", justify="left", padx=5)
        self.tokens_info_label.pack(fill="x", padx=10, pady=5)

        # شروع بارگذاری در نخ جدا
        threading.Thread(target=self.load_engine, daemon=True).start()

    def load_engine(self):
        """بارگذاری موتور LlamaIndex و به‌روزرسانی GUI پس از اتمام."""
        try:
            self.query_engine, self.token_counter, status = initialize_llama_index()
            self.status_label.config(text=status)
            self.ask_button.config(state="normal")
        except Exception as e:
            messagebox.showerror("خطای راه‌اندازی", f"خطا در هنگام راه‌اندازی LlamaIndex:\n{e}")
            self.status_label.config(text="❌ خطای راه‌اندازی")

    def start_query_thread(self):
        """شروع یک نخ جدید برای اجرای پرسش تا GUI قفل نشود."""
        query_text = self.input_entry.get()
        if not query_text.strip():
            messagebox.showwarning("ورودی خالی", "لطفاً یک سوال وارد کنید.")
            return
        
        self.ask_button.config(state="disabled")
        self.status_label.config(text="⏳ در حال پردازش سوال...")
        
        # پاک کردن خروجی‌های قبلی
        self.update_text_widget(self.output_text, "")
        self.update_text_widget(self.subq_text, "")
        self.tokens_info_label.config(text="")

        threading.Thread(target=self.run_query, args=(query_text,), daemon=True).start()

    def run_query(self, query_text):
        """اجرای واقعی پرسش و به‌روزرسانی GUI."""
        try:
            self.token_counter.reset_counts()
            response = self.query_engine.query(query_text)

            # استخراج پاسخ
            final_answer = str(response)
            self.update_text_widget(self.output_text, final_answer)

            # استخراج زیرسوال‌ها
            generated_questions = []
            for source_node in response.source_nodes:
                if "sub_question" in source_node.metadata:
                    generated_questions.append(source_node.metadata["sub_question"].strip())
            
            if generated_questions:
                sub_q_text = "\n".join(f"{i}: {q}" for i, q in enumerate(set(generated_questions), 1))
                self.update_text_widget(self.subq_text, sub_q_text)
            else:
                self.update_text_widget(self.subq_text, "هیچ زیر-سوالی تولید نشد.")

            # استخراج آمار توکن‌ها
            embedding_tokens = self.token_counter.total_embedding_token_count
            llm_tokens = self.token_counter.llm_token_count
            total_tokens = embedding_tokens + llm_tokens
            token_info = (
                f"توکن‌های Embedding: {embedding_tokens}\n"
                f"توکن‌های LLM (پرسش و پاسخ): {llm_tokens}\n"
                f"مجموع توکن‌ها: {total_tokens}"
            )
            self.tokens_info_label.config(text=token_info)

            self.status_label.config(text="✅ آماده پرسش")

        except Exception as e:
            messagebox.showerror("خطای پرسش", f"خطا در هنگام پردازش سوال:\n{e}")
            self.status_label.config(text="❌ خطای پردازش")
        
        finally:
            self.ask_button.config(state="normal")

    def update_text_widget(self, widget, text):
        """تابع کمکی برای به‌روزرسانی ویجت‌های متنی."""
        widget.config(state="normal")
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.config(state="disabled")

def main():
    """تابع اصلی برای راه‌اندازی برنامه."""
    setup_directories()
    root = tk.Tk()
    app = LlamaApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
