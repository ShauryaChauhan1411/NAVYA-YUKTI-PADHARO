import os
import re
from datetime import datetime
from transformers import pipeline
from gtts import gTTS
import pandas as pd
import matplotlib.pyplot as plt
import random

MODEL_ITINERARY = "google/flan-t5-base"
TRANSLATION_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es"
}
DASHBOARD_ROOT = "dashboards"

os.makedirs(DASHBOARD_ROOT, exist_ok=True)

def recommend_cities(interests, days):
    city_map = {
        "heritage": ["Jaipur", "Udaipur", "Jodhpur", "Chittorgarh"],
        "food": ["Jaipur", "Udaipur", "Ajmer", "Pushkar"],
        "desert": ["Jaisalmer", "Bikaner", "Barmer"],
        "adventure": ["Jaisalmer", "Mount Abu", "Kumbhalgarh"],
        "nature": ["Mount Abu", "Ranakpur", "Sariska", "Udaipur"],
        "shopping": ["Jaipur", "Pushkar", "Jodhpur"],
        "culture": ["Jaipur", "Pushkar", "Udaipur"]
    }

    selected = []
    for interest in interests:
        key = interest.strip().lower()
        for k in city_map:
            if k in key:
                selected.extend(city_map[k])

    if not selected:
        selected = ["Jaipur", "Udaipur", "Jodhpur"]

    seen, ordered = set(), []
    for c in selected:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    return ordered[:days]

def generate_itinerary(budget=12000, days=3, interests=None):
    if interests is None:
        interests = ["heritage", "food"]

    selected_cities = recommend_cities(interests, days)
    city_list = ", ".join(selected_cities)

    prompt = (
        f"You are a creative travel planner.\n"
        f"Create a fun, day-wise itinerary for a {days}-day Rajasthan trip under ₹{budget}, "
        f"focusing on {', '.join(interests)}. Cities to use: {city_list}. "
        f"Include real attractions, food, culture, and experiences. Format like:\n"
        f"Day 1 - Jaipur:\n• Visit Amber Fort\n• Try Rajasthani thali\n\n"
        f"Now write the itinerary below:\n"
    )

    print(f"\nGenerating itinerary for: {city_list} ... (this may take a while depending on environment)")
    try:
        generator = pipeline("text2text-generation", model=MODEL_ITINERARY)
        result = generator(prompt, max_new_tokens=256, num_return_sequences=1)
        raw_text = result[0].get("generated_text", "") or result[0].get("text", "")
    except Exception as e:
        print("Model generation failed:", e)
        raw_text = ""

    structured = format_itinerary(raw_text, days, budget, selected_cities)
    save_itinerary(structured)
    return structured

def format_itinerary(text, days, budget, cities):
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"•\s*•+", "•", text)
    text = re.sub(r"(?i)\bday\s*(\d+)\b", lambda m: f"Day {m.group(1)}", text)
    text = text.strip()

    if not re.search(r"Day\s*1", text, re.IGNORECASE) or len(text) < 60:
        print("Using fallback creative itinerary (AI text too short).")
        fun_activities = {
            "Jaipur": [
                "Visit Amber Fort and Hawa Mahal",
                "Shop for handicrafts in Johari Bazaar",
                "Enjoy traditional Rajasthani dinner with folk dance"
            ],
            "Udaipur": [
                "Take a boat ride on Lake Pichola",
                "Explore City Palace and Jag Mandir",
                "Watch sunset at Fateh Sagar Lake"
            ],
            "Jodhpur": [
                "Climb Mehrangarh Fort for city views",
                "Wander the Blue City lanes",
                "Dine on rooftop with view of Umaid Bhawan Palace"
            ],
            "Jaisalmer": [
                "Explore Jaisalmer Fort and Patwon Ki Haveli",
                "Ride a camel on the Sam Sand Dunes",
                "Enjoy folk music under the desert stars"
            ],
            "Mount Abu": [
                "Trek to Guru Shikhar peak",
                "Visit Dilwara Temples",
                "Boat ride on Nakki Lake"
            ],
            "Pushkar": [
                "Visit Brahma Temple",
                "Walk around Pushkar Lake ghats",
                "Shop in colorful street bazaars"
            ]
        }

        fallback_texts = []
        for i in range(days):
            city = cities[i % len(cities)]
            acts = random.sample(fun_activities.get(city, ["Explore local attractions"]), 2)
            fallback_texts.append(
                f"Day {i+1} - {city}:\n"
                f"• {acts[0]}\n"
                f"• {acts[1]}"
            )
        text = "\n\n".join(fallback_texts)

    header = f"Rajasthan Itinerary ({days} Days, ₹{budget})\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    lines = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        parts = block.split("\n")
        header_line = parts[0].strip()
        body_lines = []
        for ln in parts[1:]:
            ln = ln.strip()
            if not ln:
                continue
            if not ln.startswith("•"):
                ln = "• " + ln
            body_lines.append(ln)
        lines.append(header_line + ("\n" + "\n".join(body_lines) if body_lines else ""))

    final_text = header + "\n\n".join(lines)
    return final_text


def save_itinerary(text, filename="generated_itinerary.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nItinerary saved to: {filename}")
    except Exception as e:
        print("Could not save itinerary:", e)

def translate_description(text, target_lang="hi"):
    if target_lang not in TRANSLATION_MODELS:
        raise ValueError("Unsupported language. Choose from 'hi', 'fr', or 'es'.")
    model_name = TRANSLATION_MODELS[target_lang]
    print("\nTranslating text...")
    try:
        translator = pipeline("translation", model=model_name)
        translated = translator(text, max_new_tokens=256)
        return translated[0].get("translation_text", "") or translated[0].get("generated_text", "")
    except Exception as e:
        print("⚠️ Translation failed:", e)
        return text


def text_to_speech(text, lang="hi", filename="output.mp3"):
    print("\n Generating audio file...")
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        print(f"Audio saved as {filename}")
    except Exception as e:
        print("TTS generation failed:", e)

def generate_insights_dashboard():
    print("\nTourism Insights Dashboard")
    theme = input("Enter tourism theme (heritage/food/desert/adventure/nature/shopping): ").strip().lower()
    data_map = {
        "heritage": {
            "location": ["Jaipur", "Udaipur", "Jodhpur", "Chittorgarh"],
            "mentions": [230, 180, 140, 110],
            "sentiment": [0.90, 0.86, 0.82, 0.79]
        },
        "food": {
            "location": ["Jaipur", "Udaipur", "Ajmer", "Pushkar"],
            "mentions": [170, 140, 95, 85],
            "sentiment": [0.88, 0.85, 0.80, 0.78]
        }
    }

    if theme not in data_map:
        print("Unknown theme. Showing 'heritage' data.")
        theme = "heritage"

    data = data_map[theme]
    df = pd.DataFrame(data)
    df["tourism_score"] = (df["mentions"] * df["sentiment"]).round(2)

    folder = os.path.join(DASHBOARD_ROOT, f"{theme}_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(folder, exist_ok=True)

    csv_path = os.path.join(folder, "insights.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nSaved insights CSV: {csv_path}")

    for col, title in [("mentions", "Tourist Mentions"), ("sentiment", "Sentiment"), ("tourism_score", "Tourism Score")]:
        plt.figure(figsize=(8, 4))
        plt.bar(df["location"], df[col])
        plt.title(f"{title} by City ({theme.capitalize()})")
        plt.tight_layout()
        path = os.path.join(folder, f"{col}_chart.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved chart: {path}")

    print("\nGenerating AI summary...")
    try:
        generator = pipeline("text2text-generation", model=MODEL_ITINERARY)
        res = generator(
            f"Summarize tourism trends for {theme} in Rajasthan with key attractions and mood.",
            max_new_tokens=120
        )
        summary = res[0].get("generated_text", "")
    except Exception as e:
        print("Summary generation failed:", e)
        summary = "Tourism remains strong with positive visitor sentiment."

    with open(os.path.join(folder, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSummary saved: {summary}")

    return folder

def main():
    print("\n=== NavyaYukti AI — Smart Tourism Companion ===")
    print("1. Generate AI Itinerary")
    print("2. Translate Description")
    print("3. Generate Insights Dashboard")
    print("4. Text-to-Speech for Text File")
    print("0. Exit")

    while True:
        choice = input("\nChoose option (0/1/2/3/4): ").strip()
        if choice == "0":
            print("Goodbye ")
            break
        elif choice == "1":
            budget = int(input("Enter your budget (₹) [default 12000]: ") or 12000)
            days = int(input("Number of travel days [default 3]: ") or 3)
            interests_input = input("Enter interests (comma separated) [heritage, food]: ").strip()
            interests = [s.strip() for s in interests_input.split(",")] if interests_input else ["heritage", "food"]
            itinerary = generate_itinerary(budget=budget, days=days, interests=interests)
            print("\n Generated Itinerary:\n")
            print(itinerary)
        elif choice == "2":
            text = input("Enter English text to translate: ").strip()
            lang = input("Target language (hi/fr/es) [hi]: ").strip() or "hi"
            print(translate_description(text, lang))
        elif choice == "3":
            folder = generate_insights_dashboard()
            print(f" Dashboard saved to {folder}")
        elif choice == "4":
            path = input("Enter text file path: ").strip()
            lang = input("Audio language (en/hi/fr/es) [en]: ").strip() or "en"
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            text_to_speech(content, lang=lang)
        else:
            print("Invalid choice. Please enter 0–4.")


if __name__ == "__main__":
    main()
