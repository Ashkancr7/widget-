from docx import Document

def convert_docx_to_txt(docx_path, txt_path):
    doc = Document(docx_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for para in doc.paragraphs:
            f.write(para.text + '\n')

# 📄 مسیر فایل وردتو اینجا بذار
docx_path = "data4/laptop.docx"
txt_path = "data4/laptop.txt"

convert_docx_to_txt(docx_path, txt_path)
print("✅ فایل تبدیل شد به .txt")
