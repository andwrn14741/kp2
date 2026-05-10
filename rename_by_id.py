import os
import re

ROOT = r"C:\Users\Андрей\PycharmProjects\parser2\dataset"

def extract_id(name):
    """
    Извлекает цифры после символа №
    Пример: 'Audi ... №123055995_files' → '123055995'
    """
    m = re.search(r"№\s*(\d+)", name)
    return m.group(1) if m else None


for item in os.listdir(ROOT):
    old_path = os.path.join(ROOT, item)

    ad_id = extract_id(item)
    if not ad_id:
        print("❌ ID не найден:", item)
        continue

    # Определяем новое имя
    if os.path.isdir(old_path):
        new_name = ad_id
    else:
        # HTML-файл → сохраняем расширение
        ext = os.path.splitext(item)[1]
        new_name = f"{ad_id}{ext}"

    new_path = os.path.join(ROOT, new_name)

    # Переименовываем
    if old_path != new_path:
        os.rename(old_path, new_path)
        print(f"✔ '{item}' → '{new_name}'")

print("🎉 Готово! Папки и HTML-файлы переименованы.")