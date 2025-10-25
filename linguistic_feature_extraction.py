import os
import json
import re
import nltk
import textstat
from textblob import TextBlob
from collections import Counter
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import difflib
from nltk.corpus import stopwords

# Ensure required NLTK packages are downloaded
nltk.download("punkt")
nltk.download("stopwords")

# Paths to segmented data
DATA_DIR = "scraped_text"
FILES = {
    "human_books": os.path.join(DATA_DIR, "segmented_books.json"),
    "human_reddit": os.path.join(DATA_DIR, "segmented_reddit.json"),
    "human_wikipedia": os.path.join(DATA_DIR, "segmented_wikipedia.json"),
    "ai_generated": os.path.join(DATA_DIR, "segmented_ai.json"),
}
OUTPUT_FILE = os.path.join(DATA_DIR, "linguistic_features.json")

# AI-like phrases (commonly found in AI-generated text)
AI_PHRASES = [
    "it is important to note that", "in conclusion", "the results suggest that",
    "this study highlights", "moreover", "furthermore", "in contrast",
    "this perspective sheds light on", "based on our findings"
]

# Personal pronouns
PERSONAL_PRONOUNS = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON_PRONOUNS = {"you", "your", "yours"}

# Transition words (overused by AI)
TRANSITION_WORDS = {
    "moreover", "furthermore", "in contrast", "in conclusion",
    "therefore", "thus", "however", "on the other hand", "consequently",
    "for example", "for instance", "nevertheless", "hence", "accordingly"
}

def measure_sentence_length(text):
    """Computes mean & standard deviation of sentence length."""
    sentences = sent_tokenize(text)
    if not sentences:
        return {"mean_length": 0, "std_dev": 0}

    lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    mean_length = sum(lengths) / len(lengths)
    std_dev = (sum((x - mean_length) ** 2 for x in lengths) / len(lengths)) ** 0.5

    return {"mean_length": mean_length, "std_dev": std_dev}

def analyze_vocabulary_richness(text):
    """Computes Type-Token Ratio (TTR) for vocabulary richness."""
    words = word_tokenize(text)
    if not words:
        return {"TTR": 0}

    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    return {"TTR": ttr}

def detect_personal_pronouns(text):
    """Counts occurrences of first-person and second-person pronouns."""
    words = word_tokenize(text.lower())
    return {
        "first_person_count": sum(1 for word in words if word in PERSONAL_PRONOUNS),
        "second_person_count": sum(1 for word in words if word in SECOND_PERSON_PRONOUNS),
    }

def check_ai_like_phrasing(text):
    """Detects AI-like phrases in text (case-insensitive match)."""
    detected_phrases = set([phrase.lower() for phrase in AI_PHRASES if phrase in text.lower()])
    return {"ai_phrases_count": len(detected_phrases), "phrases": list(detected_phrases)}

def analyze_sentence_complexity(text):
    """Computes Flesch Reading Ease Score (Lower = More Complex)."""
    try:
        return {"readability_score": textstat.flesch_reading_ease(text)}
    except:
        return {"readability_score": 0}  # Avoids crashes on invalid input

def detect_repetitive_structure(text):
    """Identifies repetitive sentence structures using n-grams & similarity detection."""
    text = text.lower()
    
    # Extract 3-word n-gram patterns
    ngram_patterns = re.findall(r"(\w{3,} \w{3,} \w{3,})", text)  
    common_patterns = Counter(ngram_patterns).most_common(5)

    # Count total repetitions
    total_repetition = sum(count for _, count in common_patterns)

    # Detect sentence similarity (for AI text)
    sentences = sent_tokenize(text)
    repeated_sentences = sum(
        1 for i in range(len(sentences) - 1) 
        if difflib.SequenceMatcher(None, sentences[i], sentences[i + 1]).ratio() > 0.8
    )

    return {
        "repeated_patterns": common_patterns,
        "repetition_count": total_repetition + repeated_sentences  # Include near-duplicates
    }

def punctuation_density(text):
    """Measures punctuation frequency in text."""
    punctuation_counts = Counter(re.findall(r"[.,!?;:]", text))
    total_words = len(word_tokenize(text))

    if total_words == 0:
        return {"punctuation_density": 0}  # Prevent division by zero

    return {"punctuation_density": sum(punctuation_counts.values()) / total_words}

def stopword_usage(text):
    """Measures stopword frequency in text."""
    words = word_tokenize(text.lower())
    stopword_count = sum(1 for word in words if word in stopwords.words("english"))

    return {"stopword_ratio": stopword_count / len(words) if words else 0}

def transition_word_overuse(text):
    """Counts transition words appearing at the start OR anywhere in sentences."""
    sentences = sent_tokenize(text)
    
    transition_start_count = sum(
        1 for sent in sentences 
        if (words := word_tokenize(sent.lower())) and words[0] in TRANSITION_WORDS
    )

    transition_anywhere_count = sum(
        1 for sent in sentences 
        if any(word in TRANSITION_WORDS for word in word_tokenize(sent.lower()))
    )

    return {
        "transition_sentence_ratio": transition_start_count / len(sentences) if sentences else 0,
        "transition_anywhere_ratio": transition_anywhere_count / len(sentences) if sentences else 0
    }

def compute_semantic_coherence(text):
    """Estimates semantic coherence using TextBlob sentiment polarity."""
    try:
        return {"coherence_score": abs(TextBlob(text).sentiment.polarity)}
    except:
        return {"coherence_score": 0}  # Avoid None values

def extract_features(text, label):
    """Extracts all linguistic features from the text and assigns labels."""
    return {
        "label": label,  # Preserve label (0 = Human, 1 = AI)
        "sentence_length": measure_sentence_length(text),
        "vocabulary_richness": analyze_vocabulary_richness(text),
        "pronoun_usage": detect_personal_pronouns(text),
        "ai_phrasing": check_ai_like_phrasing(text),
        "complexity": analyze_sentence_complexity(text),
        "repetitive_structure": detect_repetitive_structure(text),
        "punctuation": punctuation_density(text),
        "stopword_usage": stopword_usage(text),
        "transition_usage": transition_word_overuse(text),
        "semantic_coherence": compute_semantic_coherence(text),
    }

def process_all_sources():
    """Processes all datasets (books, reddit, wikipedia, AI-generated)."""
    all_features = {}

    for dataset in FILES:
        if not os.path.exists(FILES[dataset]):
            print(f"⚠️ {dataset} file missing. Skipping...")
            continue

        try:
            with open(FILES[dataset], "r", encoding="utf-8") as f:
                texts = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in {dataset}. Skipping...")
            continue

        if not texts or not isinstance(texts, list):  # Validate dataset
            print(f"⚠️ Skipping {dataset}: No valid text data.")
            continue

        print(f"🛠 Extracting features from {len(texts)} texts in {dataset}...")

        label = 0 if "human" in dataset else 1  # Assign label
        feature_data = [extract_features(text, label) for text in tqdm(texts, desc=f"🔍 Processing {dataset}")]

        all_features[dataset] = feature_data

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_features, f, indent=2)

    print(f"✅ Linguistic features extracted and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_sources()
