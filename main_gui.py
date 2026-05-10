import os
import re
import glob
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from catboost import CatBoostRegressor
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QTextEdit, QFileDialog
)
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl
from PySide6.QtGui import QFont
from groq import Groq
# ============================================================
#  НАСТРОЙКИ
# ============================================================
GROQ_API_KEY = "api"
client = Groq(api_key=GROQ_API_KEY)
log_window: QTextEdit | None = None
result_window: QTextEdit | None = None
MODEL_PATH = "model_price.cbm"
# ============================================================
#  ЛОГ
# ============================================================
def log(msg: str):
    global log_window
    print(msg)
    if log_window is not None:
        log_window.append(msg)
def warn(msg: str):
    log(f"[WARN] {msg}")
def error(msg: str):
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
    log("Парсинг характеристик...")
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
    params = soup.find("div", class_="card__params")
    if params:
        text = params.get_text(" ", strip=True).lower()
        m = re.search(r"(\d{4})\s*г", text)
        if m:
            result["year"] = int(m.group(1))
        m = re.search(r"(\d,\d)\s*л", text)
        if m:
            result["engine_volume"] = float(m.group(1).replace(",", "."))
        m = re.search(r"(\d[\d\s ]*)\s*км", text)
        if m:
            result["mileage"] = int(m.group(1).replace(" ", "").replace(" ", ""))
        for g in ["механика", "автомат", "робот", "вариатор", "dsg"]:
            if g in text:
                result["gearbox"] = g
                break
        if "бензин" in text:
            result["engine_type"] = "бензин"
        if "дизель" in text:
            result["engine_type"] = "дизель"
        if "гибрид" in text:
            result["engine_type"] = "гибрид"
    desc = soup.find("div", class_="card__description")
    if desc:
        text = desc.get_text(" ", strip=True).lower()
        for b in ["седан", "универсал", "купе", "лифтбек", "хэтчбек", "другой"]:
            if b in text:
                result["body"] = b
                break
        for d in ["передний привод", "задний привод", "полный привод"]:
            if d in text:
                result["drive"] = d
                break
        for c in ["белый", "черный", "чёрный", "серый", "синий", "красный", "серебристый", "другой"]:
            if c in text:
                result["color"] = c
                break
    mod = soup.find("div", class_="card__modification")
    if mod:
        text = mod.get_text(" ", strip=True).lower()
        m = re.search(r"(\d+)\s*л\.с", text)
        if m:
            result["power_hp"] = int(m.group(1))
        m = re.search(r"расход\s*(\d+,\d+|\d+)", text)
        if m:
            result["fuel_consumption"] = float(m.group(1).replace(",", "."))
    return result
# ============================================================
#  КУРС USD НБРБ
# ============================================================
usd_rate_cache = None
def get_usd_rate():
    global usd_rate_cache
    if usd_rate_cache is not None:
        return usd_rate_cache
    try:
        url = "https://api.nbrb.by/exrates/rates/431"
        r = requests.get(url, timeout=3)
        data = r.json()
        rate = float(data["Cur_OfficialRate"])
        usd_rate_cache = rate
        log(f"[КУРС НБРБ] 1 USD = {rate} BYN")
        return rate
    except Exception as e:
        warn(f"Не удалось получить курс НБРБ: {e}, fallback 2.81")
        usd_rate_cache = 2.81
        return usd_rate_cache
# ============================================================
#  ПАРСИНГ ЦЕНЫ USD
# ============================================================
def parse_price_usd(soup):
    tag = soup.find("button", class_="card__price-button")
    if not tag:
        warn("Цена BYN не найдена")
        return None
    text = tag.get_text(" ", strip=True)
    log(f"Цена BYN raw: {text}")
    match = re.search(r'([\d\s\u00A0]+)', text)
    if not match:
        warn(f"Не удалось извлечь число из: {text}")
        return None
    price_str = re.sub(r'[\s\u00A0]', '', match.group(1))
    try:
        price_byn = int(price_str)
        log(f"Цена BYN: {price_byn}")
        usd_rate = get_usd_rate()
        price_usd = int(price_byn / usd_rate)
        log(f"Цена USD (по курсу НБРБ {usd_rate}): {price_usd}")
        return price_usd
    except Exception as e:
        warn(f"Ошибка конвертации цены: {e}")
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
    match = re.search(r'([\d\s]+)\s*\$', text)
    if match:
        price_str = re.sub(r'\s', '', match.group(1))
        try:
            price = int(price_str)
            log(f"Средняя цена USD: {price}")
            return price
        except Exception:
            pass
    warn(f"Не удалось распарсить среднюю цену USD: {text}")
    return None
# ============================================================
#  ПАРСИНГ HTML
# ============================================================
def parse_avby_soup(soup) -> dict:
    title = soup.find("title").text.strip() if soup.find("title") else None
    log(f"Заголовок: {title}")
    price_usd = parse_price_usd(soup)
    market_price = parse_market_price(soup)
    brand = model = generation = None
    breadcrumb = soup.find("div", class_="breadcrumb")
    if breadcrumb:
        items = breadcrumb.find_all("span", class_="link-text")
        if len(items) >= 2:
            brand = items[1].get_text(strip=True)
        if len(items) >= 3:
            model = items[2].get_text(strip=True)
        if len(items) >= 4:
            generation = items[3].get_text(strip=True)
    else:
        warn("Breadcrumb НЕ найден")
    desc_tag = soup.find("div", class_="card__comment-text")
    description = desc_tag.text.strip() if desc_tag else ""
    log(f"Описание: {description[:200]}...")
    char_data = parse_characteristics_block(soup)
    return {
        "title": title,
        "brand": brand,
        "model": model,
        "generation": generation,
        "price_usd": price_usd,
        "market_price_usd": market_price,
        "description": description,
        **char_data
    }
def parse_avby_html(path: str) -> dict:
    log(f"\n=== Парсинг файла: {path} ===")
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    return parse_avby_soup(soup)
def parse_avby_html_from_string(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    return parse_avby_soup(soup)
# ============================================================
#  NLP-ФЛАГИ
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
#  LLM-АНАЛИЗ (GROQ)
# ============================================================
def llm_analyze(description: str):
    if not description or len(description.strip()) < 10:
        warn("Описание слишком короткое, пропускаем LLM анализ")
        return {
            "condition_score": 50,
            "repair_risk": 50,
            "engine": "нет данных",
            "gearbox": "нет данных",
            "body": "нет данных",
            "documents": "нет данных",
            "other": "нет данных"
        }
    prompt = f"""
Ты — опытный автоэксперт, который умеет читать объявления так же, как перекупы и мастера СТО.

Твоя задача — разобрать описание автомобиля по блокам, учитывая:
- сленг
- скрытые намёки
- «ходовые» выражения
- попытки продавца скрыть проблемы
- противоречия в тексте
- реальные риски, которые стоят за фразами

Примеры интерпретации:
- "менять прокладку под башкой" → серьёзная проблема двигателя
- "есть рыжики" → коррозия, но не критично
- "по кузову есть нюансы" → скрытые дефекты
- "сел и поехал" → обычно значит «ездит, но не идеально»
- "вложений не требует" → плюс, но если есть противоречия — игнорировать
- "нужно вложить" → минус
- "ездил в такси" → повышенный износ
- "стояла долго" → риск по топливной системе и тормозам
- "обслужена" → плюс
- "новый мотор" → плюс, но возможны причины замены
- "без дисков" → минус по комплектности
- "обмен" → часто хотят избавиться от проблемного авто

Тебе нужно выдать строго JSON:

{{
  "condition_score": 0-100,
  "repair_risk": 0-100,
  "engine": "краткий вывод по двигателю",
  "gearbox": "краткий вывод по коробке",
  "body": "краткий вывод по кузову",
  "documents": "краткий вывод по документам",
  "other": "прочие важные замечания"
}}
Где:
- condition_score — итоговое состояние авто
- repair_risk — вероятность серьёзных вложений в ближайшее время
Описание:
\"\"\"{description}\"\"\"
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=512,
            stream=False
        )
        raw = completion.choices[0].message.content
        log(f"LLM ответ: {raw[:200]}...")
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start == -1 or end == 0:
            warn("Не найден JSON в ответе LLM")
            return {
                "condition_score": 50,
                "repair_risk": 50,
                "engine": "нет данных",
                "gearbox": "нет данных",
                "body": "нет данных",
                "documents": "нет данных",
                "other": "нет данных"
            }
        json_str = raw[start:end]
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        data = json.loads(json_str)
        return {
            "nlp_condition_score": int(data.get("condition_score", 50)),
            "nlp_repair_risk": int(data.get("repair_risk", 50)),
            "llm_engine": data.get("engine", "нет данных"),
            "llm_gearbox": data.get("gearbox", "нет данных"),
            "llm_body": data.get("body", "нет данных"),
            "llm_documents": data.get("documents", "нет данных"),
            "llm_other": data.get("other", "нет данных")
        }
    except Exception as e:
        warn(f"Ошибка LLM (Groq): {e}")
        return {
            "nlp_condition_score": 50,
            "nlp_repair_risk": 50,
            "llm_engine": "нет данных",
            "llm_gearbox": "нет данных",
            "llm_body": "нет данных",
            "llm_documents": "нет данных",
            "llm_other": "нет данных"
        }
# ============================================================
#  ЗАГРУЗКА МОДЕЛИ
# ============================================================
log("Загрузка модели CatBoost...")
model = CatBoostRegressor()
model.load_model(MODEL_PATH)
log("Модель загружена")
# ============================================================
#  АНАЛИЗ HTML ФАЙЛА
# ============================================================
def analyze_html_file(html_path: str, result_window):
    log("\n====================================")
    log(f"Анализ HTML файла: {html_path}")
    log("====================================\n")
    parsed_data = parse_avby_html(html_path)
    if not parsed_data["brand"] or not parsed_data["model"]:
        error("Не удалось определить марку и модель автомобиля")
        show_error_result("Не удалось определить марку и модель автомобиля. Проверь HTML.", result_window)
        return
    text = parsed_data.get("description") or ""
    flags = nlp_flags(text)
    deep = llm_analyze(text)
    price_usd = parsed_data.get("price_usd", 0) or 0
    market_price = parsed_data.get("market_price_usd", 0) or 0
    price_gap = market_price - price_usd if market_price and price_usd else 0
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
    show_analysis_result(parsed_data, flags, deep, price_pred, price_usd, market_price, result_window)
def analyze_html_string(html, result_window):
    parsed_data = parse_avby_html_from_string(html)
    if not parsed_data.get("brand") or not parsed_data.get("model"):
        show_error_result("Не удалось определить марку и модель автомобиля.", result_window)
        return
    text = parsed_data.get("description") or ""
    flags = nlp_flags(text)
    deep = llm_analyze(text)
    price_usd = parsed_data.get("price_usd", 0) or 0
    market_price = parsed_data.get("market_price_usd", 0) or 0
    price_gap = market_price - price_usd if market_price and price_usd else 0
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
    price_pred = model.predict(df_model)[0]
    show_analysis_result(parsed_data, flags, deep, price_pred, price_usd, market_price, result_window)
# ============================================================
#  ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА
# ============================================================
# ============================================================
#  HTML-ЦВЕТА
# ============================================================
def color_green(text):
    return f"<span style='color:#00aa00; font-weight:bold;'>{text}</span>"
def color_yellow(text):
    return f"<span style='color:#d4a017; font-weight:bold;'>{text}</span>"
def color_red(text):
    return f"<span style='color:#cc0000; font-weight:bold;'>{text}</span>"
def title_big(text):
    return f"<span style='font-size:18px; font-weight:bold;'>{text}</span>"
def section(text):
    return f"<span style='font-size:14px; font-weight:bold; margin-top:10px;'>{text}</span>"
# ============================================================
#  ОШИБКА
# ============================================================
def show_error_result(msg: str):
    global result_window
    result_window.clear()
    result_window.append(color_red(f"[ОШИБКА] {msg}"))
# ============================================================
#  ОСНОВНАЯ КАРТОЧКА АНАЛИЗА
# ============================================================
def show_analysis_result(parsed_data, flags, deep, price_pred, price_usd, market_price, result_window):
    result_window.clear()
    title = f"{parsed_data.get('brand','')} {parsed_data.get('model','')} {parsed_data.get('generation','')}".strip()
    # ============================================================
    # 1. БАЗОВЫЕ РАСЧЁТЫ
    # ============================================================
    diff = 0
    diff_percent = 0.0
    if price_usd > 0:
        diff = price_pred - price_usd
        diff_percent = (diff / price_usd) * 100
    if price_usd > 0 and market_price:
        diff_market_pct = ((market_price - price_usd) / price_usd) * 100
    else:
        diff_market_pct = 0
    # ============================================================
    # 2. ОПРЕДЕЛЕНИЕ СЕРЬЁЗНЫХ НЕИСПРАВНОСТЕЙ (НОВАЯ ЛОГИКА)
    # ============================================================
    text_engine = deep['llm_engine'].lower()
    text_gearbox = deep['llm_gearbox'].lower()
    text_body = deep['llm_body'].lower()
    text_docs = deep['llm_documents'].lower()
    text_other = deep['llm_other'].lower()
    # --- ДВИГАТЕЛЬ ---
    serious_engine = any(k in text_engine for k in [
        "газы", "гбц", "трес", "перегрев", "масложор", "дым", "стук", "стуки",
        "тнвд", "насос", "компресс", "компрессия", "не тянет", "не заводится"
    ])
    # --- КОРОБКА ---
    serious_gearbox = any(k in text_gearbox for k in [
        "пробуксов", "пинает", "рывк", "аварий", "течь", "гул", "не включается",
        "выбивает", "сцепление буксует"
    ])
    # --- КУЗОВ ---
    serious_body = any(k in text_body for k in [
        "гнил", "ржав", "сквозная", "днище", "пороги", "стойк", "варить",
        "сгнил", "сильная коррозия"
    ])
    # --- ДОКУМЕНТЫ ---
    serious_docs = any(k in text_docs for k in [
        "арест", "ограничен", "запрет", "дубликат", "утерян", "розыск", "залог"
    ])
    # --- ПРОЧЕЕ ---
    serious_other = any(k in text_other for k in [
        "стук", "стуки", "не работает", "не функционирует", "под замену",
        "после долгого простоя", "неисправ", "сломано"
    ])
    # Итог
    serious_problems = serious_engine or serious_gearbox or serious_body or serious_docs or serious_other
    # ============================================================
    # 3. НОВАЯ ЛОГИКА ВЕРДИКТА
    # ============================================================
    if serious_problems:
        # 1) Цена ниже рынка на 40%+ → можно брать как проект
        if diff_market_pct >= 40:
            final_verdict = color_yellow("⚠ МОЖНО РАССМАТРИВАТЬ ТОЛЬКО КАК ПРОЕКТ / ПОД ВЛОЖЕНИЯ")
            final_reason = f"Цена ниже рынка на {diff_market_pct:.0f}%, но есть серьёзные неисправности."
        # 2) Цена ниже рынка на 20–40% → проект, но осторожно
        elif 20 <= diff_market_pct < 40:
            final_verdict = color_yellow("⚠ МОЖНО РАССМАТРИВАТЬ ТОЛЬКО КАК ПРОЕКТ / ПОД ВЛОЖЕНИЯ")
            final_reason = f"Цена ниже рынка на {diff_market_pct:.0f}%, но состояние требует вложений."
        # 3) Цена ниже рынка <20% → не рекомендую
        else:
            final_verdict = color_red("❌ НЕ РЕКОМЕНДУЮ")
            final_reason = "Есть серьёзные неисправности, а цена недостаточно низкая."
    else:
        # НЕТ серьёзных неисправностей
        if diff_market_pct >= 20:
            final_verdict = color_green("ОТЛИЧНАЯ СДЕЛКА")
            final_reason = f"Цена ниже рынка на {diff_market_pct:.0f}%."
        elif diff_market_pct >= 5:
            final_verdict = color_green("ХОРОШАЯ СДЕЛКА")
            final_reason = f"Цена ниже рынка на {diff_market_pct:.0f}%."
        elif -5 <= diff_market_pct < 5:
            final_verdict = color_yellow("НОРМА")
            final_reason = "Цена близка к рыночной."
        elif -15 <= diff_market_pct < -5:
            final_verdict = color_yellow("ПЕРЕПЛАТА, НО МОЖНО РАССМАТРИВАТЬ")
            final_reason = f"Цена выше рынка на {abs(diff_market_pct):.0f}%, но состояние хорошее."
        else:
            final_verdict = color_red("НЕ РЕКОМЕНДУЮ")
            final_reason = f"Цена выше рынка на {abs(diff_market_pct):.0f}%."
    # ============================================================
    # 4. ВЫВОД В GUI
    # ============================================================
    result_window.append(title_big(title) + "<br><br>")
    result_window.append(f"{final_verdict}<br>")
    result_window.append(f"{final_reason}<br><br>")
    # ---------------- ЦЕНЫ ----------------
    result_window.append(section("💰 Цены") + "<br>")
    result_window.append(f"- Оценка модели: {int(price_pred)} $<br>")
    if price_usd:
        result_window.append(f"- Цена в объявлении: {price_usd} $<br>")
    if market_price:
        result_window.append(f"- Средняя цена: {market_price} $<br>")
    if price_usd:
        result_window.append(f"- Отклонение: {diff:+.0f} $ ({diff_percent:+.1f}%)<br><br>")
    # ---------------- ТЕХНИЧЕСКИЕ ДАННЫЕ ----------------
    result_window.append(section("⚙ Технические данные") + "<br>")
    result_window.append(f"- Год: {parsed_data.get('year')}<br>")
    result_window.append(f"- Двигатель: {parsed_data.get('engine_volume')} л, {parsed_data.get('engine_type')}<br>")
    result_window.append(f"- Мощность: {parsed_data.get('power_hp')} л.с.<br>")
    result_window.append(f"- Коробка: {parsed_data.get('gearbox')}<br>")
    result_window.append(f"- Привод: {parsed_data.get('drive')}<br>")
    result_window.append(f"- Кузов: {parsed_data.get('body')}<br>")
    result_window.append(f"- Цвет: {parsed_data.get('color')}<br>")
    result_window.append(f"- Расход: {parsed_data.get('fuel_consumption')} л/100 км<br>")
    result_window.append(f"- Пробег: {parsed_data.get('mileage')} км<br><br>")
    # ---------------- NLP ФЛАГИ ----------------
    result_window.append(section("🧠 Анализ текста") + "<br>")
    result_window.append(f"- Оценка состояния (LLM): {deep['nlp_condition_score']}/100<br>")
    result_window.append(f"- Риск ремонтов (LLM): {deep['nlp_repair_risk']}/100<br><br>")
    # ---------------- LLM РАЗБОР ----------------
    result_window.append(section("🔍 Разбор описания (LLM)") + "<br>")
    result_window.append(f"<b>Двигатель:</b> {deep['llm_engine']}<br>")
    result_window.append(f"<b>Коробка:</b> {deep['llm_gearbox']}<br>")
    result_window.append(f"<b>Кузов:</b> {deep['llm_body']}<br>")
    result_window.append(f"<b>Документы:</b> {deep['llm_documents']}<br>")
    result_window.append(f"<b>Прочее:</b> {deep['llm_other']}<br>")
# ============================================================
#  GUI
# ============================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AV.BY Анализатор HTML (CatBoost + Groq)")
        # ---------------- КНОПКИ ----------------
        btn_open = QPushButton("Выбрать HTML файл")
        btn_open.setFont(QFont("Arial", 12))
        btn_open.clicked.connect(self.choose_file)

        btn_analyze_live = QPushButton("Анализировать текущую страницу")
        btn_analyze_live.setFont(QFont("Arial", 12))
        btn_analyze_live.clicked.connect(self.analyze_live_page)
        # ---------------- БРАУЗЕР ----------------
        self.browser = QWebEngineView()
        self.browser.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.browser.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.browser.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        self.browser.settings().setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        self.browser.settings().setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        self.browser.load(QUrl("https://av.by"))
        # ---------------- ОКНО РЕЗУЛЬТАТОВ ----------------
        self.result_window = QTextEdit()
        self.result_window.setReadOnly(True)
        self.result_window.setFont(QFont("Consolas", 10))
        self.result_window.setAcceptRichText(True)
        # ---------------- ЛЕЙАУТ ----------------
        h = QHBoxLayout()
        h.addWidget(self.browser, stretch=64)
        h.addWidget(self.result_window, stretch=36)
        v = QVBoxLayout()
        v.addWidget(btn_open)
        v.addWidget(btn_analyze_live)
        v.addLayout(h)
        self.setLayout(v)
        self.resize(1400, 800)
    # --------------------------------------------------------
    # АНАЛИЗ ТЕКУЩЕЙ СТРАНИЦЫ — ПРАВИЛЬНОЕ МЕСТО
    # --------------------------------------------------------
    def analyze_live_page(self):
        def callback(html):
            if html:
                analyze_html_string(html, self.result_window)
            else:
                self.result_window.setText("Не удалось получить HTML со страницы")

        self.browser.page().runJavaScript(
            "document.documentElement.outerHTML;",
            callback
        )
    # --------------------------------------------------------
    # Выбор HTML файла
    # --------------------------------------------------------
    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать HTML файл",
            "",
            "HTML файлы (*.html *.htm);;Все файлы (*.*)"
        )
        if file_path:
            self.browser.load(QUrl.fromLocalFile(file_path))
            analyze_html_file(file_path, self.result_window)
# ============================================================
# ТОЧКА ВХОДА
# ============================================================
if __name__ == "__main__":
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--enable-gpu --ignore-gpu-blocklist"
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())