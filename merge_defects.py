import glob
import pandas as pd
import re


# =========================
# Нормализация модели
# =========================
def normalize_model(m: str) -> str:
    if not isinstance(m, str):
        return ""
    m = m.lower().strip()

    # убираем год
    m = re.sub(r"\b(19|20)\d{2}\b", "", m)

    # убираем скобки
    m = re.sub(r"\([^)]*\)", "", m)

    # убираем тип кузова/лишнее
    for w in ["седан", "универсал", "variant", "комби", "хэтчбек", "wagon"]:
        m = m.replace(w, "")

    # убираем двойные пробелы
    m = re.sub(r"\s+", " ", m)

    return m.strip()


# =========================
# Загрузка всех CSV болячек
# =========================
files = glob.glob(r"C:\Users\Андрей\PycharmProjects\parser2\cars_defects\*.csv")

dfs = []
for f in files:
    try:
        df = pd.read_csv(f, sep=",", engine="python", on_bad_lines="skip")
        df["brand"] = df["brand"].str.lower().str.strip()
        df["model"] = df["model"].apply(normalize_model)
        dfs.append(df)
        print(f"Загружено: {f} ({len(df)} строк)")
    except Exception as e:
        print(f"Ошибка загрузки {f}: {e}")


# =========================
# Объединение в один DataFrame
# =========================
df_all = pd.concat(dfs, ignore_index=True)

print("\nВсего строк после объединения:", len(df_all))
print("Уникальных брендов:", df_all["brand"].nunique())
print("Уникальных моделей:", df_all["model"].nunique())


# =========================
# Сохранение итогового файла
# =========================
output_path = "all_defects_flat.csv"
df_all.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nГотово! Итоговый файл сохранён как: {output_path}")