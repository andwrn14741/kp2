import os
import re
import glob
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from catboost import CatBoostRegressor
import sys
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                               QHBoxLayout, QTextEdit, QFileDialog)
from PySide6.QtGui import QFont

# ============================================================
#  ЛОГ
# ============================================================

log_window = None
result_window = None


def log(msg):
    global log_window
    print(msg)
    if log_window is not None:
        log_window.insert("end", msg + "\n")
        log_window.see("end")


def warn(msg):
    log(f"[WARN] {msg}")


def error(msg):
    log(f"[ERROR] {msg}")


# ============================================================
#  НОРМАЛИЗАЦИЯ МОДЕЛИ
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
#  ПАРСИНГ ХАРАКТЕРИСТИК
# ============================================================

def parse_characteristics_block(soup):
    log("Парсинг характеристик (card__params / card__description / card__modification)...")

    result = {
        "year": None,
        "gearbox": None,
        "engine_volume": None,
        "engine_type": None,
        "mileage": None,
        "body": None,
        "drive": None,
        "color": None,
        "power_hp": None,
        "fuel_consumption": None,
    }

    # ---------------------------------------------------------
    # 1. card__params
    # ---------------------------------------------------------
    params = soup.find("div", class_="card__params")
    if params:
        text = params.get_text(" ", strip=True).lower()
        log(f"  card__params найден: {text}")

        # год
        m = re.search(r"(\d{4})\s*г", text)
        if m:
            result["year"] = int(m.group(1))
            log(f"    Год выпуска: {result['year']}")

        # коробка
        gearbox_variants = [
            "механика", "автомат", "робот", "вариатор",
            "dsg", "tiptronic", "multitronic", "двухсцепление"
        ]
        for g in gearbox_variants:
            if g in text:
                result["gearbox"] = g
                log(f"    Коробка: {g}")
                break

        # объём двигателя
        m = re.search(r"(\d,\d)\s*л", text)
        if m:
            result["engine_volume"] = float(m.group(1).replace(",", "."))
            log(f"    Объём двигателя: {result['engine_volume']} л")

        # топливо
        fuel_variants = [
            "бензин", "дизель", "гибрид", "электро",
            "газ", "метан", "пропан", "бензин (метан)", "бензин (газ)"
        ]
        for f in fuel_variants:
            if f in text:
                result["engine_type"] = f
                log(f"    Тип двигателя: {f}")
                break

        # пробег
        m = re.search(r"(\d[\d\s ]*)\s*км", text)
        if m:
            result["mileage"] = int(m.group(1).replace(" ", "").replace(" ", ""))
            log(f"    Пробег: {result['mileage']} км")
    else:
        warn("  card__params НЕ найден")

    # ---------------------------------------------------------
    # 2. card__description
    # ---------------------------------------------------------
    desc = soup.find("div", class_="card__description")
    if desc:
        text = desc.get_text(" ", strip=True).lower()
        log(f"  card__description найден: {text}")

        # кузов
        body_variants = [
            "внедорожник 3 дв.", "внедорожник 5 дв.",
            "кабриолет", "купе", "легковой фургон",
            "лимузин", "лифтбек",
            "микроавтобус грузопассажир", "микроавтобус пассажирский",
            "пикап", "родстер", "седан", "универсал",
            "хэтчбек 3 дв.", "хэтчбек 5 дв.",
            "другой"
        ]
        for b in body_variants:
            if b in text:
                result["body"] = b
                log(f"    Кузов: {b}")
                break

        # привод
        drive_variants = [
            "передний привод",
            "задний привод",
            "подключаемый полный привод",
            "постоянный полный привод",
        ]
        for d in drive_variants:
            if d in text:
                result["drive"] = d
                log(f"    Привод: {d}")
                break

        # цвет
        color_variants = [
            "белый", "бордовый", "жёлтый", "желтый",
            "зелёный", "зеленый", "коричневый",
            "красный", "оранжевый", "серебристый",
            "серый", "синий", "фиолетовый",
            "чёрный", "черный", "другой"
        ]
        for c in color_variants:
            if c in text:
                result["color"] = c
                log(f"    Цвет: {c}")
                break
    else:
        warn("  card__description НЕ найден")

    # ---------------------------------------------------------
    # 3. card__modification
    # ---------------------------------------------------------
    mod = soup.find("div", class_="card__modification")
    if mod:
        text = mod.get_text(" ", strip=True).lower()
        log(f"  card__modification найден: {text}")

        # мощность
        m = re.search(r"(\d+)\s*л\.с", text)
        if m:
            result["power_hp"] = int(m.group(1))
            log(f"    Мощность: {result['power_hp']} л.с.")

        # расход
        m = re.search(r"расход\s*(\d+,\d+|\d+)", text)
        if m:
            result["fuel_consumption"] = float(m.group(1).replace(",", "."))
            log(f"    Расход: {result['fuel_consumption']} л/100км")
    else:
        warn("  card__modification НЕ найден")

    return result


# ============================================================
#  ПАРСИНГ ЦЕНЫ USD
# ============================================================

def parse_price_usd(soup):
    # Ищем кнопку с ценой в BYN
    tag = soup.find("button", class_="card__price-button")
    if not tag:
        warn("Цена BYN не найдена")
        return None

    text = tag.get_text(" ", strip=True)
    log(f"Цена BYN raw: {text}")

    # Извлекаем число из формата "11 256 р." или "11&nbsp;256&nbsp;р."
    # Убираем все разделители (пробелы, &nbsp;, нецифровые символы) кроме цифр
    match = re.search(r'([\d\s\u00A0]+)', text)
    if match:
        price_str = re.sub(r'[\s\u00A0]', '', match.group(1))  # убираем пробелы и &nbsp;
        try:
            price_byn = int(price_str)
            log(f"Цена BYN: {price_byn}")

            # Конвертация BYN → USD
            usd_to_byn_rate = 2.81
            price_usd = int(price_byn / usd_to_byn_rate)
            log(f"Цена USD (конвертация): {price_usd}")

            return price_usd
        except:
            pass

    warn(f"Не удалось распарсить цену BYN: {text}")
    return None


# ============================================================
#  ПАРСИНГ СРЕДНЕЙ ЦЕНЫ USD
# ============================================================

def parse_market_price(soup):
    block = soup.find("div", class_="featured__price-value")
    if not block:
        warn("Средняя цена не найдена")
        return None

    text = block.get_text(" ", strip=True)
    log(f"Средняя цена raw: {text}")

    # Ищем цену в долларах (формат: "8 844 $")
    match = re.search(r'([\d\s]+)\s*\$', text)
    if match:
        price_str = re.sub(r'[\s]', '', match.group(1))
        try:
            price = int(price_str)
            log(f"Средняя цена USD: {price}")
            return price
        except:
            pass

    warn(f"Не удалось распарсить среднюю цену USD: {text}")
    return None


# ============================================================
#  ОСНОВНОЙ ПАРСЕР HTML
# ============================================================

def parse_avby_html(path: str) -> dict:
    log(f"\n=== Парсинг файла: {path} ===")

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    # Заголовок
    title = soup.find("title").text.strip() if soup.find("title") else None
    log(f"Заголовок: {title}")

    # Цена
    price_usd = parse_price_usd(soup)

    # Средняя цена
    market_price = parse_market_price(soup)

    # ==========================================================
    #  BREADCRUMB
    # ==========================================================

    brand = None
    model = None
    generation = None

    breadcrumb = soup.find("div", class_="breadcrumb")
    if breadcrumb:
        items = breadcrumb.find_all("span", class_="link-text")
        log(f"Breadcrumb найден: {len(items)} элементов")

        if len(items) >= 2:
            brand = items[1].get_text(strip=True)
            log(f"  Бренд: {brand}")

        if len(items) >= 3:
            model = items[2].get_text(strip=True)
            log(f"  Модель: {model}")

        if len(items) >= 4:
            generation = items[3].get_text(strip=True)
            log(f"  Поколение: {generation}")
    else:
        warn("Breadcrumb НЕ найден")

    # Описание
    desc_tag = soup.find("div", class_="card__comment-text")
    description = desc_tag.text.strip() if desc_tag else None
    log(f"Описание: {description}")

    # Характеристики
    char_data = parse_characteristics_block(soup)

    log("Файл успешно разобран")

    return {
        "file": os.path.basename(path),
        "title": title,
        "brand": brand,
        "model": model,
        "generation": generation,
        "price_usd": price_usd,
        "market_price_usd": market_price,
        "description": description,
        **char_data
    }


# ============================================================
#  NLP-ФЛАГИ (как в train_model.py)
# ============================================================

def nlp_flags(text: str) -> dict:
    if text is None:
        text = ""
    t = text.lower()

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
#  LLM-АНАЛИЗ
# ============================================================

def llm_analyze(description: str):
    if not description or len(description.strip()) < 10:
        warn("Описание слишком короткое, пропускаем LLM анализ")
        return {
            "nlp_condition_score": 50,
            "nlp_repair_risk": 50
        }

    prompt = f"""
Ты — эксперт по диагностике автомобилей.
Проанализируй описание объявления и оцени состояние авто.

Верни строго JSON без лишнего текста:

{{"condition_score": 0-100, "repair_risk": 0-100}}

Описание:
{description}
"""

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "t-tech/T-lite-it-2.1:q4_k_m",
                "stream": False
            },
            timeout=30
        )
        raw = r.json()["response"]
        log(f"LLM ответ: {raw[:200]}...")

        # Ищем JSON в ответе
        start = raw.find('{')
        end = raw.rfind('}') + 1

        if start == -1 or end == 0:
            warn("Не найден JSON в ответе LLM")
            return {
                "nlp_condition_score": 50,
                "nlp_repair_risk": 50
            }

        json_str = raw[start:end]
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        data = json.loads(json_str)

        return {
            "nlp_condition_score": int(data.get("condition_score", 50)),
            "nlp_repair_risk": int(data.get("repair_risk", 50))
        }

    except json.JSONDecodeError as e:
        warn(f"Ошибка парсинга JSON: {e}")
        return {
            "nlp_condition_score": 50,
            "nlp_repair_risk": 50
        }
    except requests.exceptions.RequestException as e:
        warn(f"Ошибка запроса к Ollama: {e}")
        return {
            "nlp_condition_score": 50,
            "nlp_repair_risk": 50
        }
    except Exception as e:
        warn(f"Неизвестная ошибка LLM: {e}")
        return {
            "nlp_condition_score": 50,
            "nlp_repair_risk": 50
        }


# ============================================================
#  ЗАГРУЗКА БОЛЯЧЕК
# ============================================================

DEFECTS_DIR = r"C:\Users\Андрей\PycharmProjects\parser2\cars_defects"


def load_defects():
    files = glob.glob(os.path.join(DEFECTS_DIR, "*.csv"))
    if not files:
        warn("Папка с болячками не найдена или пуста")
        return pd.DataFrame()
    dfs = []
    for f in files:
        d = pd.read_csv(f, sep=",", engine="python", on_bad_lines="skip")
        d["brand"] = d["brand"].str.lower()
        d["model"] = d["model"].apply(normalize_model)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


df_defects = load_defects()


def match_year(year, years_str):
    if not isinstance(years_str, str):
        return True
    m = re.match(r"(\d{4})-(\d{4})", years_str)
    if not m:
        return True
    y1, y2 = int(m.group(1)), int(m.group(2))
    return y1 <= int(year) <= y2


def find_defects(brand, model, year):
    if not brand or not model or not year or df_defects.empty:
        return None

    subset = df_defects[
        (df_defects["brand"] == brand.lower()) &
        (df_defects["model"] == normalize_model(model))
        ]

    subset = subset[subset["years"].apply(lambda y: match_year(year, y))]

    if subset.empty:
        return None

    cost_mean = subset["cost_mean_usd"].mean()
    prob_mean = subset["issue_probability_pct"].mean()
    impact_mean = subset["issue_price_impact_pct"].mean()
    risk_mean = subset["model_risk_score"].mean()
    gen_cost_mean = subset["generation_avg_cost_usd"].mean()

    expected_cost = cost_mean * (prob_mean / 100)
    total_risk = (prob_mean / 100) * (impact_mean / 100) * risk_mean

    return {
        "defect_cost_mean_usd": cost_mean,
        "defect_issue_probability_pct": prob_mean,
        "defect_price_impact_pct": impact_mean,
        "defect_model_risk_score": risk_mean,
        "defect_generation_avg_cost_usd": gen_cost_mean,
        "defect_expected_cost_usd": expected_cost,
        "defect_total_risk_score": total_risk,
    }


# ============================================================
#  ЗАГРУЗКА МОДЕЛИ
# ============================================================

log("Загрузка модели...")
model = CatBoostRegressor()
model.load_model("model_price.cbm")
log("Модель загружена")


# ============================================================
#  АНАЛИЗ HTML ФАЙЛА
# ============================================================

def analyze_html_file(html_path: str):
    log(f"\n====================================")
    log(f"Анализ HTML файла: {html_path}")
    log(f"====================================\n")

    # Парсим HTML
    parsed_data = parse_avby_html(html_path)

    # Проверяем, что данные получены
    if not parsed_data["brand"] or not parsed_data["model"]:
        error("Не удалось определить марку и модель автомобиля")
        show_error_result("Не удалось определить марку и модель автомобиля. Проверьте структуру HTML файла.")
        return

    # Болячки
    defects = find_defects(parsed_data["brand"], parsed_data["model"], parsed_data["year"])
    if defects is None:
        defects = {
            "defect_cost_mean_usd": 0.0,
            "defect_issue_probability_pct": 0.0,
            "defect_price_impact_pct": 0.0,
            "defect_model_risk_score": 0.0,
            "defect_generation_avg_cost_usd": 0.0,
            "defect_expected_cost_usd": 0.0,
            "defect_total_risk_score": 0.0,
        }

    # NLP и LLM
    text = parsed_data.get("description") or ""
    flags = nlp_flags(text)
    deep = llm_analyze(text)

    price_usd = parsed_data.get("price_usd", 0) or 0
    market_price = parsed_data.get("market_price_usd", 0) or 0
    price_gap = market_price - price_usd if market_price else 0

    log(f"Цена из объявления: {price_usd} $")
    log(f"Средняя цена: {market_price} $")

    # Формируем строку для модели
    row = {
        "year": parsed_data.get("year", 0) or 0,
        "mileage": parsed_data.get("mileage", 0) or 0,
        "engine_volume": parsed_data.get("engine_volume", 0) or 0,
        "engine_type": parsed_data.get("engine_type", "unknown") or "unknown",
        "gearbox": parsed_data.get("gearbox", "unknown") or "unknown",
        "drive": parsed_data.get("drive", "unknown") or "unknown",
        "body": parsed_data.get("body", "unknown") or "unknown",
        "color": parsed_data.get("color", "unknown") or "unknown",
        "power_hp": parsed_data.get("power_hp", 0) or 0,
        "fuel_consumption": parsed_data.get("fuel_consumption", 0) or 0,
        "price_usd": price_usd,
        "market_price_usd": market_price,
        "price_gap": price_gap,

        **defects,
        **flags,

        "nlp_condition_score": deep["nlp_condition_score"],
        "nlp_repair_risk": deep["nlp_repair_risk"],
    }

    df_model = pd.DataFrame([row])

    for col in df_model.columns:
        if df_model[col].dtype == "O":
            df_model[col] = df_model[col].fillna("unknown").astype(str)
        else:
            df_model[col] = df_model[col].fillna(0)

    log("Отправка данных в модель...")
    price_pred = model.predict(df_model)[0]
    log(f"Прогноз модели: {int(price_pred)} $")

    # Выводим результат
    show_analysis_result(parsed_data, defects, flags, deep, price_pred, price_usd, market_price)


# ============================================================
#  ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА (исправленная версия с процентами)
# ============================================================

def show_analysis_result(parsed_data, defects, flags, deep, price_pred, price_usd, market_price):
    global result_window

    result_window.delete(1.0, "end")

    title = f"{parsed_data.get('brand', '')} {parsed_data.get('model', '')} {parsed_data.get('generation', '')}".strip()
    result_window.insert("end", f"{title}\n", "title")

    # Расчет в процентах
    if price_usd > 0:
        diff_percent = ((price_pred - price_usd) / price_usd) * 100
        diff = price_pred - price_usd
        savings = diff if diff > 0 else 0
        overpay = -diff if diff < 0 else 0

        log(f"Разница в цене: {diff:+.0f} $ ({diff_percent:+.1f}%)")
    else:
        diff_percent = 0
        diff = 0
        savings = 0
        overpay = 0
        log("Цена в объявлении не найдена, использую только оценку состояния")

    # ============================================================
    #  НОВЫЙ ВЕРДИКТ С УЧЕТОМ ВСЕХ ФАКТОРОВ
    # ============================================================

    # Расчет процентов
    if price_usd > 0:
        diff_from_model_percent = ((price_pred - price_usd) / price_usd) * 100  # +/- от цены объявления
        diff_from_market_percent = ((market_price - price_usd) / price_usd) * 100 if market_price else 0  # +/- от рынка

        log(f"Отклонение от модели: {diff_from_model_percent:+.1f}%")
        log(f"Отклонение от рынка: {diff_from_market_percent:+.1f}%")

        # Логика вердикта:
        # 1. Если цена сильно выше рынка (>15%) - переплата
        if diff_from_market_percent < -15:
            verdict = "ПЕРЕПЛАТА"
            verdict_color = "red"
            verdict_icon = "💰"
            reason = f"Цена на {abs(diff_from_market_percent):.0f}% выше средней по рынку"

        # 2. Если цена сильно ниже рынка (>15%) - отличная сделка
        elif diff_from_market_percent > 15:
            verdict = "ОТЛИЧНАЯ СДЕЛКА"
            verdict_color = "green"
            verdict_icon = "✅"
            reason = f"Цена на {diff_from_market_percent:.0f}% ниже средней по рынку"

        # 3. Если модель показывает переоценку >20% относительно цены объявления
        elif diff_from_model_percent < -20:
            verdict = "ЗАВЫШЕНА"
            verdict_color = "orange"
            verdict_icon = "⚠️"
            reason = f"Модель оценивает авто на {abs(diff_from_model_percent):.0f}% дешевле"

        # 4. Если модель показывает выгоду >15%
        elif diff_from_model_percent > 15:
            verdict = "ВЫГОДНО"
            verdict_color = "green"
            verdict_icon = "👍"
            reason = f"Модель оценивает авто на {diff_from_model_percent:.0f}% дороже"

        # 5. Нормальный диапазон ±10%
        elif -10 <= diff_from_model_percent <= 10:
            verdict = "НОРМА"
            verdict_color = "yellow"
            verdict_icon = "📊"
            reason = "Цена соответствует рыночной"

        else:
            verdict = "С ОСТОРОЖНОСТЬЮ"
            verdict_color = "orange"
            verdict_icon = "⚠️"
            reason = "Требуется личный осмотр"

        # Оценка цены
        if diff_from_market_percent > 15:
            price_rating = f"ОТЛИЧНАЯ ЦЕНА (ниже рынка на {diff_from_market_percent:.0f}%) 📈"
            price_color = "green"
        elif diff_from_market_percent > 5:
            price_rating = f"Хорошая цена (ниже рынка на {diff_from_market_percent:.0f}%) 📈"
            price_color = "green"
        elif -5 <= diff_from_market_percent <= 5:
            price_rating = f"Средняя цена ({diff_from_market_percent:+.0f}% от рынка) 📊"
            price_color = "yellow"
        elif diff_from_market_percent > -15:
            price_rating = f"Завышена (выше рынка на {abs(diff_from_market_percent):.0f}%) 📉"
            price_color = "orange"
        else:
            price_rating = f"СИЛЬНО ЗАВЫШЕНА (выше рынка на {abs(diff_from_market_percent):.0f}%) 📉📉"
            price_color = "red"

    # Вывод результатов
    result_window.insert("end", f"\n{verdict_icon} ВЕРДИКТ: {verdict}\n", verdict_color)
    if price_usd > 0:
        result_window.insert("end", f"{'=' * 50}\n", "normal")
        result_window.insert("end", f"💰 {price_rating}\n", price_color)
    result_window.insert("end", f"{'=' * 50}\n\n", "normal")

    if savings > 0 and price_usd > 0:
        result_window.insert("end", f"💎 ПОТЕНЦИАЛЬНАЯ ЭКОНОМИЯ: {savings:,.0f} $ ({diff_percent:.0f}%)\n", "green")
    elif overpay > 0 and price_usd > 0:
        result_window.insert("end", f"⚠️ ПЕРЕПЛАТА: {overpay:,.0f} $ ({abs(diff_percent):.0f}%)\n", "orange")
    elif price_usd > 0:
        result_window.insert("end", f"💰 Цена близка к рыночной ({diff_percent:+.0f}%)\n", "yellow")

    result_window.insert("end", f"{'=' * 50}\n\n", "normal")

    # Вывод чисел
    result_window.insert("end", f"Оценка модели: {int(price_pred):,} $\n".replace(",", " "))
    if price_usd > 0:
        result_window.insert("end", f"Цена в объявлении: {price_usd:,} $\n".replace(",", " "))
    if market_price:
        result_window.insert("end", f"Средняя цена: {market_price:,} $\n".replace(",", " "))
    if price_usd > 0:
        result_window.insert("end", f"Отклонение: {diff:+,} $ ({diff_percent:+.1f}%)\n\n".replace(",", " "))
    else:
        result_window.insert("end", f"Цена в объявлении не найдена\n\n")

    # Остальная информация без изменений...
    result_window.insert("end", "⚙ Технические данные\n", "section")
    result_window.insert("end", f"- Год: {parsed_data.get('year')}\n")
    result_window.insert("end",
                         f"- Двигатель: {parsed_data.get('engine_volume')} л, {parsed_data.get('engine_type')}\n")
    result_window.insert("end", f"- Мощность: {parsed_data.get('power_hp')} л.с.\n")
    result_window.insert("end", f"- Коробка: {parsed_data.get('gearbox')}\n")
    result_window.insert("end", f"- Привод: {parsed_data.get('drive')}\n")
    result_window.insert("end", f"- Кузов: {parsed_data.get('body')}\n")
    result_window.insert("end", f"- Цвет: {parsed_data.get('color')}\n")
    result_window.insert("end", f"- Расход: {parsed_data.get('fuel_consumption')} л/100 км\n")
    if parsed_data.get('mileage'):
        result_window.insert("end", f"- Пробег: {parsed_data.get('mileage'):,} км\n\n".replace(",", " "))
    else:
        result_window.insert("end", f"- Пробег: не указан\n\n")

    result_window.insert("end", "🛠 Болячки модели\n", "section")

    if defects['defect_total_risk_score'] > 0.5:
        risk_color = "red"
        risk_icon = "🔴"
    elif defects['defect_total_risk_score'] > 0.2:
        risk_color = "yellow"
        risk_icon = "🟡"
    else:
        risk_color = "green"
        risk_icon = "🟢"

    result_window.insert("end", f"{risk_icon} Риск болячек: ", risk_color)
    result_window.insert("end", f"{defects['defect_total_risk_score']:.3f}\n")
    result_window.insert("end", f"- Средняя стоимость ремонта: {defects['defect_cost_mean_usd']:.0f} $\n")
    result_window.insert("end", f"- Вероятность проблемы: {defects['defect_issue_probability_pct']:.1f} %\n")
    result_window.insert("end", f"- Ожидаемые расходы: {defects['defect_expected_cost_usd']:.0f} $\n\n")

    result_window.insert("end", "🧠 Анализ описания\n", "section")

    if deep['nlp_condition_score'] > 70:
        condition_color = "green"
        condition_text = "ХОРОШЕЕ"
    elif deep['nlp_condition_score'] > 40:
        condition_color = "yellow"
        condition_text = "СРЕДНЕЕ"
    else:
        condition_color = "red"
        condition_text = "ПЛОХОЕ"

    result_window.insert("end", f"📊 Состояние: {condition_text} ({deep['nlp_condition_score']}/100)\n",
                         condition_color)

    if deep['nlp_repair_risk'] > 60:
        repair_color = "red"
        repair_text = "ВЫСОКИЙ"
    elif deep['nlp_repair_risk'] > 30:
        repair_color = "yellow"
        repair_text = "СРЕДНИЙ"
    else:
        repair_color = "green"
        repair_text = "НИЗКИЙ"

    result_window.insert("end", f"🔧 Риск ремонта: {repair_text} ({deep['nlp_repair_risk']}/100)\n\n", repair_color)

    result_window.insert("end", f"- Двигатель: {'⚠ проблема' if flags['nlp_engine_bad'] else '✅ OK'}\n")
    result_window.insert("end", f"- Коробка: {'⚠ проблема' if flags['nlp_gearbox_bad'] else '✅ OK'}\n")
    result_window.insert("end", f"- Кузов: {'⚠ проблема' if flags['nlp_body_bad'] else '✅ OK'}\n")
    result_window.insert("end", f"- Обслуживание: {'✅ хорошее' if flags['nlp_service_good'] else '⚠ нейтрально'}\n")
    result_window.insert("end", f"- Такси: {'⚠ риск' if flags['nlp_taxi_risk'] else '✅ нет'}\n")
    result_window.insert("end", f"- Хранение: {'⚠ риск' if flags['nlp_storage_risk'] else '✅ нет'}\n\n")

    result_window.insert("end", "📝 ИТОГОВАЯ РЕКОМЕНДАЦИЯ\n", "section")
    result_window.insert("end", f"{reason}\n\n", verdict_color)

    if savings > 500 and price_usd > 0:
        result_window.insert("end", f"💎 ПОТЕНЦИАЛЬНАЯ ЭКОНОМИЯ: {savings:,.0f} $ ({diff_percent:.0f}%)\n", "green")
    elif savings > 0 and price_usd > 0:
        result_window.insert("end", f"💰 Экономия: {savings:,.0f} $ ({diff_percent:.0f}%)\n", "green")
    elif overpay > 0 and price_usd > 0:
        result_window.insert("end", f"⚠️ ПЕРЕПЛАТА: {overpay:,.0f} $ ({abs(diff_percent):.0f}%)\n", "orange")


# ============================================================
#  ВЫБОР ФАЙЛА
# ============================================================

def select_and_analyze_file():
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Выберите HTML файл объявления",
        "",
        "HTML Files (*.html *.htm);;All Files (*)"
    )

    if file_path:
        analyze_html_file(file_path)


# ============================================================
#  АДАПТЕР ДЛЯ ПРИЛОЖЕНИЯ
# ============================================================

class TkTextAdapter:
    def __init__(self, widget):
        self.widget = widget
        self.styles = {}

    def insert(self, _, text, tag=None):
        if tag and tag in self.styles:
            style = self.styles[tag]
            if isinstance(text, str):
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html = f'<span style="{style}">{text}</span>'
                self.widget.append(html)
            else:
                self.widget.append(str(text))
        else:
            self.widget.append(text)

    def delete(self, start, end):
        self.widget.clear()

    def see(self, index):
        pass

    def tag_config(self, tag, **kwargs):
        style = []
        if "foreground" in kwargs:
            color_map = {
                "green": "#00aa00",
                "red": "#cc0000",
                "yellow": "#ccaa00",
                "orange": "#ff6600",
                "black": "#000000"
            }
            color = kwargs["foreground"]
            if color in color_map:
                style.append(f"color:{color_map[color]}")
            else:
                style.append(f"color:{color}")
        if "font" in kwargs:
            font = kwargs["font"]
            if isinstance(font, tuple):
                if "bold" in font:
                    style.append("font-weight:bold")
        self.styles[tag] = ";".join(style)


# ============================================================
#  ОСНОВНОЕ ОКНО
# ============================================================

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AV.BY Анализатор HTML")
        self.resize(1400, 800)

        layout = QVBoxLayout()

        self.btn = QPushButton("Выбрать HTML файл объявления")
        self.btn.setFont(QFont("Arial", 14))
        self.btn.clicked.connect(select_and_analyze_file)

        layout.addWidget(self.btn)

        content = QHBoxLayout()

        global log_window, result_window

        log_qt = QTextEdit()
        log_qt.setFont(QFont("Consolas", 9))
        log_qt.setReadOnly(True)

        result_qt = QTextEdit()
        result_qt.setFont(QFont("Consolas", 10))
        result_qt.setReadOnly(True)

        log_window = TkTextAdapter(log_qt)
        result_window = TkTextAdapter(result_qt)

        content.addWidget(log_qt)
        content.addWidget(result_qt)

        layout.addLayout(content)
        self.setLayout(layout)

        result_window.tag_config("title", font=("Consolas", 14, "bold"))
        result_window.tag_config("section", font=("Consolas", 11, "bold"))
        result_window.tag_config("green", foreground="green")
        result_window.tag_config("yellow", foreground="yellow")
        result_window.tag_config("red", foreground="red")
        result_window.tag_config("orange", foreground="orange")
        result_window.tag_config("normal", foreground="black")


# ============================================================
#  ЗАПУСК
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())