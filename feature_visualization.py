import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend for saving plots

# Paths
DATA_DIR = "scraped_text"
FEATURES_FILE = os.path.join(DATA_DIR, "linguistic_features.json")
OUTPUT_DIR = "feature_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define categories for comparison
HUMAN_SOURCES = ["human_books", "human_reddit", "human_wikipedia"]
AI_SOURCE = "ai_generated"

def load_data():
    """Loads extracted linguistic features from JSON."""
    if not os.path.exists(FEATURES_FILE):
        print(f"⚠️ {FEATURES_FILE} not found. Run feature extraction first.")
        return None
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_values(data, feature_key, sub_key=None):
    """Extracts values for a given feature from human & AI datasets."""
    human_values, ai_values = [], []

    for source in HUMAN_SOURCES:
        if source in data and isinstance(data[source], list):
            if sub_key:
                values = [entry[feature_key].get(sub_key, None) for entry in data[source] if isinstance(entry, dict) and feature_key in entry]
            else:
                values = [entry.get(feature_key, None) for entry in data[source] if isinstance(entry, dict) and feature_key in entry]
            human_values.extend([v for v in values if v is not None])  

    if AI_SOURCE in data and isinstance(data[AI_SOURCE], list):
        if sub_key:
            ai_values = [entry[feature_key].get(sub_key, None) for entry in data[AI_SOURCE] if isinstance(entry, dict) and feature_key in entry]
        else:
            ai_values = [entry.get(feature_key, None) for entry in data[AI_SOURCE] if isinstance(entry, dict) and feature_key in entry]
        ai_values = [v for v in ai_values if v is not None]  

    print(f"📊 Debug: Extracted {feature_key} - Human: {len(human_values)}, AI: {len(ai_values)}")

    return human_values, ai_values

def plot_distribution(human_values, ai_values, title, xlabel, filename):
    """Plots the distribution of a feature comparing human & AI text."""
    if not human_values or not ai_values:
        print(f"⚠️ Skipping {title} - No valid data available.")
        return
    
    # Add jitter to avoid overlapping points in visualization
    human_jittered = [x + np.random.normal(0, 0.05) for x in human_values]
    ai_jittered = [x + np.random.normal(0, 0.05) for x in ai_values]
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(human_jittered, label="Human", fill=True, alpha=0.3, color="blue")
    sns.kdeplot(ai_jittered, label="AI", fill=True, alpha=0.3, color="red")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    print(f"✅ Saved: {filepath}")
    plt.close()

def plot_bar_chart(human_values, ai_values, title, xlabel, ylabel, filename, use_sum=True):
    """Plots a bar chart for categorical comparison. Uses sum() for pronouns, count() for labels."""
    if not human_values or not ai_values:
        print(f"⚠️ Skipping {title} - No valid data available.")
        return

    # Use count for labels, sum for pronoun counts
    if use_sum:
        human_value = sum(human_values)  
        ai_value = sum(ai_values)  
    else:
        human_value = len(human_values)  # Count occurrences (for labels)
        ai_value = len(ai_values)

    print(f"📊 Debug: Plotting {title} - Human: {human_value}, AI: {ai_value}")

    categories = ["Human", "AI"]
    values = [human_value, ai_value]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=categories, y=values, palette=["blue", "red"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    print(f"✅ Saved: {filepath}")
    plt.close()

def visualize_features(data):
    """Generates plots comparing AI and human texts."""
    print("📊 Generating feature visualizations...")

    # Label Distribution (Use count)
    human_values, ai_values = extract_values(data, "label")
    plot_bar_chart(human_values, ai_values, "Dataset Label Distribution", "Text Source", "Count", "label_distribution.png", use_sum=False)

    # Sentence Length Variation
    human_values, ai_values = extract_values(data, "sentence_length", "mean_length")
    plot_distribution(human_values, ai_values, "Sentence Length Variation", "Mean Sentence Length", "sentence_length.png")

    # Vocabulary Richness (TTR)
    human_values, ai_values = extract_values(data, "vocabulary_richness", "TTR")
    plot_distribution(human_values, ai_values, "Vocabulary Richness (TTR)", "Type-Token Ratio (TTR)", "vocabulary_richness.png")

    # Readability Score (Flesch)
    human_values, ai_values = extract_values(data, "complexity", "readability_score")
    plot_distribution(human_values, ai_values, "Readability Score Comparison", "Flesch Reading Ease Score", "readability.png")

    # AI Phrase Detection
    human_values, ai_values = extract_values(data, "ai_phrasing", "ai_phrases_count")
    plot_distribution(human_values, ai_values, "AI Phrase Usage", "Number of AI-like Phrases", "ai_phrases.png")

    # Pronoun Usage (Use sum)
    human_values, ai_values = extract_values(data, "pronoun_usage", "first_person_count")
    plot_bar_chart(human_values, ai_values, "First-Person Pronoun Usage", "Text Source", "Total Count", "first_person_pronouns.png", use_sum=True)

    human_values, ai_values = extract_values(data, "pronoun_usage", "second_person_count")
    plot_bar_chart(human_values, ai_values, "Second-Person Pronoun Usage", "Text Source", "Total Count", "second_person_pronouns.png", use_sum=True)

    # Punctuation Density
    human_values, ai_values = extract_values(data, "punctuation", "punctuation_density")
    plot_distribution(human_values, ai_values, "Punctuation Density", "Punctuation per Word", "punctuation_density.png")

    # Stopword Usage
    human_values, ai_values = extract_values(data, "stopword_usage", "stopword_ratio")
    plot_distribution(human_values, ai_values, "Stopword Usage", "Stopword Ratio", "stopword_usage.png")

    # Transition Word Overuse
    human_values, ai_values = extract_values(data, "transition_usage", "transition_sentence_ratio")
    plot_distribution(human_values, ai_values, "Transition Word Overuse", "Proportion of Sentences with Transition Words", "transition_usage.png")

    # Semantic Coherence
    human_values, ai_values = extract_values(data, "semantic_coherence", "coherence_score")
    plot_distribution(human_values, ai_values, "Semantic Coherence Score", "Logical Coherence Score", "semantic_coherence.png")

    print("✅ All feature visualizations saved.")


if __name__ == "__main__":
    data = load_data()
    if data:
        visualize_features(data)
