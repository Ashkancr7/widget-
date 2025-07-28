from dotenv import load_dotenv, find_dotenv
import os

# سعی می‌کنیم فایل .env رو پیدا کنیم و بارگذاری کنیم
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
    print(f"✅ Loaded .env file from: {dotenv_path}")
else:
    print("⚠️ .env file not found!")

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"✅ API Key is: {api_key}")
else:
    print("❌ API Key not found in environment variables!")
