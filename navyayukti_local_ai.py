from transformers import pipeline
from gtts import gTTS
import pandas as pd
import matplotlib.pyplot as plt
import json

def generate_itinerary(budget=12000, days=3, interests=None):
    if interests is None:
        interests = ["heritage", "food"]

    prompt = f"Create a {days}-day Rajasthan itinerary under ₹{budget}, focusing on {', '.join(interests)}."
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    result = generator(prompt, max_length=512, num_return_sequences=1)
    itinerary = result[0]["generated_text"]
    return itinerary


def translate_description(text, target_lang="hi"):
    model_map = {
        "hi": "Helsinki-NLP/opus-mt-en-hi",
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "es": "Helsinki-NLP/opus-mt-en-es"
    }
    if target_lang not in model_map:
        raise ValueError("Unsupported language. Choose from 'hi', 'fr', or 'es'.")
    translator = pipeline("translation", model=model_map[target_lang])
    translated = translator(text, max_length=512)
    return translated[0]["translation_text"]


def text_to_speech(text, lang="hi", filename="output.mp3"):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    print(f" Audio saved as {filename}")


def generate_mock_insights():
    data = {
        "location": ["Jaipur", "Jodhpur", "Udaipur", "Jaisalmer", "Pushkar"],
        "mentions": [120, 90, 110, 80, 60],
        "sentiment": [0.85, 0.78, 0.82, 0.74, 0.80]
    }
    df = pd.DataFrame(data)
    print("\nTourism Insights Summary:\n", df)
    plt.figure(figsize=(6, 3))
    plt.bar(df["location"], df["sentiment"], color="orange")
    plt.title("Average Tourist Sentiment by City")
    plt.ylabel("Sentiment Score (0-1)")
    plt.tight_layout()
    plt.savefig("tourism_insights_chart.png")
    print(" Chart saved as tourism_insights_chart.png")


def main():
    print("\n=== NavyaYukti AI — Smart Tourism Companion ===")
    print("1. Generate AI Itinerary")
    print("2. Translate Monument Description")
    print("3. Generate Tourism Insights Dashboard")
    choice = input("Choose option (1/2/3): ")

    if choice == "1":
        itinerary = generate_itinerary()
        print("\nGenerated Itinerary:\n", itinerary)

    elif choice == "2":
        text = input("Enter English description: ")
        lang = input("Translate to (hi/fr/es): ")
        translation = translate_description(text, lang)
        print(f"\nTranslated text ({lang}):\n{translation}")
        tts_choice = input("Generate audio file? (y/n): ")
        if tts_choice.lower() == "y":
            text_to_speech(translation, lang, f"tour_guide_{lang}.mp3")

    elif choice == "3":
        generate_mock_insights()

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
