import os
import re
from datetime import datetime
from transformers import pipeline
from gtts import gTTS
import pandas as pd
import matplotlib.pyplot as plt
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
            if k in key and k not in selected:
                selected.extend(city_map[k])
    if not selected:
        selected = ["Jaipur", "Udaipur", "Jodhpur"]
    seen = set()
    ordered = []
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
        f"You are a friendly travel planner.\n"
        f"Create a clear, day-wise, actionable itinerary for a {days}-day trip to Rajasthan "
        f"under ₹{budget}, focusing on {', '.join(interests)}. Use only these cities: {city_list}.\n"
        f"Provide 2–3 bullet activities per day. Use the format:\n\n"
        f"Day 1 - Jaipur:\n"
        f"• Activity 1\n"
        f"• Activity 2\n\n"
        f"Day 2 - Udaipur:\n"
        f"• Activity 1\n"
        f"• Activity 2\n\n"
        f"Now write the itinerary for the requested cities/days below:\n"
    )

    print(f"\nGenerating itinerary for: {city_list} ... (this may take 15–40s depending on environment)")
    try:
        generator = pipeline("text2text-generation", model=MODEL_ITINERARY)
        result = generator(prompt, max_new_tokens=256, num_return_sequences=1)
        raw_text = result[0].get("generated_text", "") or result[0].get("text", "")
    except Exception as e:
        print(" Model generation failed:", e)
        raw_text = ""

    structured = format_itinerary(raw_text, days, budget, selected_cities)
    save_itinerary(structured)
    return structured


def format_itinerary(text, days, budget, cities):
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"•\s*•+", "•", text)
    text = re.sub(r"(?i)\bday\s*(\d+)\b", lambda m: f"Day {m.group(1)}", text)
    text = text.strip()
    if not re.search(r"Day\s*1", text, re.IGNORECASE) or len(text) < 40:
        fallback_texts = []
        for i in range(days):
            city = cities[i % len(cities)]
            fallback_texts.append(
                f"Day {i+1} - {city}:\n"
                f"• Explore major landmarks of {city} (forts, palaces or lakes)\n"
                f"• Try local cuisine and visit the main market"
            )
        text = "\n\n".join(fallback_texts)
    header = f"Rajasthan Itinerary ({days} Days, ₹{budget})\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"

    lines = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue

        block = re.sub(r"^(Day\s*\d+\s*-?\s*)(.*)$", lambda m: f"{m.group(1).strip()} {m.group(2).strip()}", block)
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
        translated_text = translated[0].get("translation_text", "") or translated[0].get("generated_text", "")
        return translated_text
    except Exception as e:
        print("Translation failed:", e)
        return text


def text_to_speech(text, lang="hi", filename="output.mp3"):
    print("\nGenerating audio file...")
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
        },
        "desert": {
            "location": ["Jaisalmer", "Bikaner", "Barmer"],
            "mentions": [160, 120, 70],
            "sentiment": [0.86, 0.81, 0.75]
        },
        "adventure": {
            "location": ["Jaisalmer", "Mount Abu", "Kumbhalgarh"],
            "mentions": [110, 100, 75],
            "sentiment": [0.83, 0.80, 0.77]
        },
        "nature": {
            "location": ["Mount Abu", "Ranakpur", "Sariska", "Udaipur"],
            "mentions": [125, 95, 85, 105],
            "sentiment": [0.86, 0.84, 0.80, 0.85]
        },
        "shopping": {
            "location": ["Jaipur", "Pushkar", "Jodhpur"],
            "mentions": [190, 120, 100],
            "sentiment": [0.89, 0.82, 0.81]
        }
    }

    if theme not in data_map:
        print("Unknown theme. Showing generic 'heritage' data.")
        theme = "heritage"

    data = data_map[theme]
    df = pd.DataFrame(data)
    df["tourism_score"] = (df["mentions"] * df["sentiment"]).round(2)
    folder = os.path.join(DASHBOARD_ROOT, f"{theme}_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, "insights.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nSaved insights CSV: {csv_path}")
    print("\nTop results:\n", df)
    plt.figure(figsize=(8, 4))
    plt.bar(df["location"], df["mentions"])
    plt.title(f"Tourist Mentions by City ({theme.capitalize()})")
    plt.ylabel("Mentions (approx.)")
    plt.tight_layout()
    mentions_chart = os.path.join(folder, "mentions_chart.png")
    plt.savefig(mentions_chart)
    plt.close()
    print(f"Saved mentions chart: {mentions_chart}")
    plt.figure(figsize=(8, 4))
    plt.bar(df["location"], df["sentiment"])
    plt.title(f"Average Sentiment by City ({theme.capitalize()})")
    plt.ylabel("Sentiment (0–1)")
    plt.tight_layout()
    sentiment_chart = os.path.join(folder, "sentiment_chart.png")
    plt.savefig(sentiment_chart)
    plt.close()
    print(f"Saved sentiment chart: {sentiment_chart}")
    plt.figure(figsize=(8, 4))
    plt.bar(df["location"], df["tourism_score"])
    plt.title(f"Tourism Score Index ({theme.capitalize()})")
    plt.ylabel("Score (mentions × sentiment)")
    plt.tight_layout()
    score_chart = os.path.join(folder, "tourism_score_chart.png")
    plt.savefig(score_chart)
    plt.close()
    print(f"Saved tourism score chart: {score_chart}")
    print("\nGenerating AI Insight Summary...")
    try:
        generator = pipeline("text2text-generation", model=MODEL_ITINERARY)
        scene = f"Theme: {theme}. Top cities: {', '.join(df['location'])}."
        insight_prompt = (
            f"Write a short 3-line tourism insight for Rajasthan based on the following: {scene} "
            f"Use plain language and mention key attractions and travel mood."
        )
        res = generator(insight_prompt, max_new_tokens=120, num_return_sequences=1)
        summary = res[0].get("generated_text", "").strip() or res[0].get("text", "")
    except Exception as e:
        print("AI insight generation failed:", e)
        summary = (
            f"{', '.join(df['location'])} are trending for {theme} tourism. "
            f"Visitors enjoy major attractions and local experiences. Sentiment is generally positive."
        )

    summary_path = os.path.join(folder, "tourism_insight_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nAI Tourism Insight Summary:\n{summary}")
    print(f"Saved summary: {summary_path}")

    return folder
def main():
    print("\n=== NavyaYukti AI — Smart Tourism Companion ===")
    print("1. Generate AI Itinerary")
    print("2. Translate Monument Description")
    print("3. Generate Tourism Insights Dashboard")
    print("4. Text-to-Speech for text file")
    print("0. Exit")

    while True:
        choice = input("\nChoose option (0/1/2/3/4): ").strip()
        if choice == "0":
            print("Goodbye ")
            break

        if choice == "1":
            try:
                budget_str = input("Enter your budget (₹) [default 12000]: ").strip()
                budget = int(budget_str) if budget_str else 12000
            except ValueError:
                budget = 12000
            try:
                days_str = input("Number of travel days [default 3]: ").strip()
                days = int(days_str) if days_str else 3
            except ValueError:
                days = 3
            interests_input = input("Enter interests (comma separated) [heritage, food]: ").strip()
            if not interests_input:
                interests = ["heritage", "food"]
            else:
                interests = [s.strip() for s in interests_input.split(",") if s.strip()]

            itinerary = generate_itinerary(budget=budget, days=days, interests=interests)
            print("\nGenerated Itinerary:\n")
            print(itinerary)

        elif choice == "2":
            text = input("Enter English description (single line or paste): ").strip()
            if not text:
                print("No text provided.")
                continue
            lang = input("Translate to (hi/fr/es) [hi]: ").strip().lower() or "hi"
            try:
                translated = translate_description(text, lang)
                print(f"\nTranslated text ({lang}):\n{translated}")
                save_choice = input("Save translated text to file? (y/n) [y]: ").strip().lower() or "y"
                if save_choice == "y":
                    fname = f"translated_{lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(translated)
                    print(f"Saved translation to {fname}")
                    tts_choice = input("Also generate audio file? (y/n) [n]: ").strip().lower() or "n"
                    if tts_choice == "y":
                        text_to_speech(translated, lang=lang, filename=f"{fname}.mp3")
            except Exception as e:
                print("Error during translation:", e)

        elif choice == "3":
            folder = generate_insights_dashboard()
            print(f"\nDashboard and charts saved in folder: {folder}")

        elif choice == "4":
            path = input("Enter path to text file to convert to speech: ").strip()
            if not os.path.isfile(path):
                print("File not found.")
                continue
            lang = input("Audio language code (hi/en/fr/es) [en]: ").strip() or "en"
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                print("File is empty.")
                continue
            out_name = f"{os.path.splitext(os.path.basename(path))[0]}_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            text_to_speech(content, lang=lang, filename=out_name)

        else:
            print("Invalid choice. Please enter 0,1,2,3 or 4.")


if __name__ == "__main__":
    main()
