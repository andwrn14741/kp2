import os
import glob
import re
import json
import requests
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. Пути
# ============================================================

CARS_CSV = "cars_dataset.csv"
DEFECTS_DIR = r"C:\Users\Андрей\PycharmProjects\parser2\cars_defects"
OUTPUT_DATASET = "cars_dataset_with_defects.csv"
MODEL_PATH = "model_price.cbm"


# ============================================================
# 2. Нормализация модели
# ============================================================

def normalize_model(m: str) -> str:
    if not isinstance(m, str):
        return ""
    m = m.lower().strip()
    m = re.sub(r"\b(19|20)\d{2}\b", "", m)
    m = re.sub(r"\([^)]*\)", "", m)
    for w in ["седан", "универсал", "variant", "комби", "хэтчбек", "wagon"]:
        m = m.replace(w, "")
    m = re.sub(r"\s+", " ", m)
    return m.strip()


# ============================================================
# 3. LLM-анализ описания
# ============================================================

def llm_analyze(description: str):
    """
    Возвращает:
    - nlp_condition_score (0–100)
    - nlp_repair_risk (0–100)
    """

    prompt = f"""
Ты — эксперт по диагностике автомобилей.
Проанализируй описание объявления и оцени состояние авто.

Верни строго JSON:

{{
  "condition_score": 0-100,
  "repair_risk": 0-100
}}

Описание:
{description}
"""

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "t-tech/T-lite-it-2.1:q4_k_m",
                "prompt": prompt,
                "stream": False
            }
        )
        raw = r.json()["response"]

        json_str = raw[raw.index("{"): raw.rindex("}")+1]
        data = json.loads(json_str)

        return {
            "nlp_condition_score": int(data.get("condition_score", 50)),
            "nlp_repair_risk": int(data.get("repair_risk", 50))
        }

    except Exception:
        return {
            "nlp_condition_score": 50,
            "nlp_repair_risk": 50
        }


# ============================================================
# 4. NLP-флаги (быстрые)
# ============================================================

def nlp_flags(text: str) -> dict:
    t = text.lower() if isinstance(text, str) else ""

    return {
        "nlp_engine_bad": int(any(k in t for k in ["стук", "дымит", "масложор", "троит"])),
        "nlp_engine_fixed": int(any(k in t for k in ["капремонт", "перебрали", "новый мотор"])),
        "nlp_gearbox_bad": int(any(k in t for k in ["пинается", "рывки", "пробуксовка", "аварийный режим"])),
        "nlp_gearbox_fixed": int(any(k in t for k in ["ремонт акпп", "новая акпп", "перебрали коробку"])),
        "nlp_body_bad": int(any(k in t for k in ["гниль", "ржавчина", "дырка", "сгнил"])),
        "nlp_body_fixed": int(any(k in t for k in ["переварен", "варили", "заменены пороги"])),
        "nlp_service_good": int(any(k in t for k in ["обслужен", "все фильтры", "масло менялось"])),
        "nlp_service_bad": int(any(k in t for k in ["требует ремонта", "нужно вложить"])),
        "nlp_storage_risk": int(any(k in t for k in ["стояла год", "долго стояла"])),
        "nlp_taxi_risk": int(any(k in t for k in ["такси", "яндекс", "bolt", "убер"])),
    }


# ============================================================
# 5. Загрузка датасета
# ============================================================

df = pd.read_csv(CARS_CSV)

required_cols = [
    "brand", "model", "year",
    "mileage", "engine_volume", "engine_type",
    "gearbox", "drive", "body", "color",
    "power_hp", "fuel_consumption",
    "price_usd", "market_price_usd"
]

if "description" not in df.columns:
    df["description"] = ""

df = df[required_cols + ["description"]].dropna(subset=required_cols)

df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].apply(normalize_model)
df["description"] = df["description"].fillna("").astype(str).str.lower()


# ============================================================
# 6. NLP-флаги + LLM-фичи
# ============================================================

nlp_cols = [
    "nlp_engine_bad", "nlp_engine_fixed",
    "nlp_gearbox_bad", "nlp_gearbox_fixed",
    "nlp_body_bad", "nlp_body_fixed",
    "nlp_service_good", "nlp_service_bad",
    "nlp_storage_risk", "nlp_taxi_risk",
    "nlp_condition_score", "nlp_repair_risk"
]

for c in nlp_cols:
    df[c] = 0

print("Анализ описаний через LLM...")

for idx in tqdm(df.index):
    text = df.at[idx, "description"]

    # быстрые флаги
    flags = nlp_flags(text)
    for k, v in flags.items():
        df.at[idx, k] = v

    # глубокий анализ
    deep = llm_analyze(text)
    df.at[idx, "nlp_condition_score"] = deep["nlp_condition_score"]
    df.at[idx, "nlp_repair_risk"] = deep["nlp_repair_risk"]


# ============================================================
# 7. Загрузка болячек
# ============================================================

defects_list = []
for f in glob.glob(os.path.join(DEFECTS_DIR, "*.csv")):
    d = pd.read_csv(f, sep=",", engine="python", on_bad_lines="skip")
    d["brand"] = d["brand"].str.lower()
    d["model"] = d["model"].apply(normalize_model)
    defects_list.append(d)

df_defects = pd.concat(defects_list, ignore_index=True)


def match_year(year, years_str):
    if pd.isna(year):
        return False
    if not isinstance(years_str, str):
        return True
    m = re.match(r"(\d{4})-(\d{4})", years_str)
    if not m:
        return True
    y1, y2 = int(m.group(1)), int(m.group(2))
    return y1 <= int(year) <= y2


def find_defects(brand, model, year):
    subset = df_defects[
        (df_defects["brand"] == brand) &
        (df_defects["model"] == model)
    ]
    if subset.empty:
        return None

    subset = subset[subset["years"].apply(lambda y: match_year(year, y))]
    if subset.empty:
        return None

    return {
        "defect_cost_mean_usd": subset["cost_mean_usd"].mean(),
        "defect_issue_probability_pct": subset["issue_probability_pct"].mean(),
        "defect_price_impact_pct": subset["issue_price_impact_pct"].mean(),
        "defect_model_risk_score": subset["model_risk_score"].mean(),
        "defect_generation_avg_cost_usd": subset["generation_avg_cost_usd"].mean(),
        "defect_expected_cost_usd": subset["cost_mean_usd"].mean() * (subset["issue_probability_pct"].mean() / 100),
        "defect_total_risk_score": (
            (subset["issue_probability_pct"].mean() / 100) *
            (subset["issue_price_impact_pct"].mean() / 100) *
            subset["model_risk_score"].mean()
        ),
    }


# ============================================================
# 8. Добавляем болячки
# ============================================================

defect_features = [
    "defect_cost_mean_usd",
    "defect_issue_probability_pct",
    "defect_price_impact_pct",
    "defect_model_risk_score",
    "defect_generation_avg_cost_usd",
    "defect_expected_cost_usd",
    "defect_total_risk_score"
]

for c in defect_features:
    df[c] = 0.0

print("Добавляем болячки...")

for idx in tqdm(df.index):
    d = find_defects(df.at[idx, "brand"], df.at[idx, "model"], df.at[idx, "year"])
    if d:
        for k, v in d.items():
            df.at[idx, k] = v


# ============================================================
# 9. Добавляем price_gap
# ============================================================

df["price_gap"] = df["market_price_usd"] - df["price_usd"]


# ============================================================
# 10. Финальные фичи
# ============================================================

features = [
    "year", "mileage", "engine_volume",
    "engine_type", "gearbox", "drive", "body", "color",
    "power_hp", "fuel_consumption",
    "price_usd", "market_price_usd", "price_gap",
] + defect_features + nlp_cols

X = df[features]
y = df["price_usd"]

cat_features = ["engine_type", "gearbox", "drive", "body", "color"]


# ============================================================
# 11. Сохраняем датасет
# ============================================================

df.to_csv(OUTPUT_DATASET, index=False)
print("Сохранено:", OUTPUT_DATASET)


# ============================================================
# 12. Обучение модели
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

try:
    print("Пробуем GPU...")
    model = CatBoostRegressor(
        iterations=2000,
        depth=8,
        learning_rate=0.02,
        loss_function="RMSE",
        task_type="GPU",
        devices="0",
        od_type="None"
    )
    print("GPU используется")
except:
    print("GPU недоступен, CPU")
    model = CatBoostRegressor(
        iterations=2000,
        depth=8,
        learning_rate=0.02,
        loss_function="RMSE",
        task_type="CPU",
        od_type="None"
    )

model.fit(train_pool, eval_set=test_pool, verbose=100)
model.save_model(MODEL_PATH)

print("Модель сохранена:", MODEL_PATH)

y_pred = model.predict(test_pool)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
