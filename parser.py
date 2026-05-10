import os
import re
import glob
import pandas as pd
from bs4 import BeautifulSoup

# ============================================================
#  ЛОГИРОВАНИЕ
# ============================================================

def log(msg):
    print(f"[LOG] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")


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

    text = text.replace("~", "").replace("$", "").strip()
    text = re.sub(r"[^\d ]", "", text)

    try:
        price = int(text.replace(" ", ""))
        log(f"Средняя цена USD: {price}")
        return price
    except:
        error(f"Ошибка парсинга средней цены: {text}")
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
#  ПАРСИНГ ВСЕЙ ПАПКИ
# ============================================================

def parse_folder(folder: str) -> pd.DataFrame:
    log(f"\n=== Парсинг папки: {folder} ===")

    files = glob.glob(os.path.join(folder, "*.html")) + glob.glob(os.path.join(folder, "*.txt"))
    log(f"Найдено файлов: {len(files)}")

    rows = []
    for f in files:
        try:
            rows.append(parse_avby_html(f))
        except Exception as e:
            error(f"Ошибка в файле {f}: {e}")

    log("Парсинг папки завершён")
    return pd.DataFrame(rows)


# ============================================================
#  ЗАПУСК
# ============================================================

if __name__ == "__main__":
    INPUT_FOLDER = "dataset"
    OUTPUT_CSV = "cars_dataset.csv"

    df = parse_folder(INPUT_FOLDER)
    df.to_csv(OUTPUT_CSV, encoding="utf-8", index=False)

    log(f"\nГотово! Собрано записей: {len(df)}")
    log(f"Сохранено в: {OUTPUT_CSV}")