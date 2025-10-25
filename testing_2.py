import pandas as pd
import joblib
import numpy as np
import fitz  # PyMuPDF for PDF text extraction
from detectgpt_roberta_prep import compute_log_probability_openvino, detect_ai_openvino_roberta
from linguistic_feature_extraction import extract_features

MODEL_PATH = r"C:\Users\Ernie\Documents\GitHub\AI_Detector\scraped_text\random_forest_ai_detector.pkl"

# ✅ Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    exit(1)  # Stop execution if model fails to load


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text.strip()if text else None
    except Exception as e:
        print(f"❌ Error Extracting Text from PDF: {e}")
        return None


def chunk_text(text, max_length=512, overlap=50):
    """Splits text into overlapping chunks for AI detection."""
    words = text.split()
    chunks = [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length - overlap)]
    return chunks


def flatten_features(feature_dict):
    """Converts nested features into a single-level dictionary while removing unwanted keys."""
    flat_dict = {}
    exclude_keys = {"label", "ai_phrasing_phrases", "repetitive_structure_repeated_patterns"}  # Keys to ignore

    for main_key, sub_dict in feature_dict.items():
        if isinstance(sub_dict, dict):  # If nested, flatten it
            for sub_key, value in sub_dict.items():
                new_key = f"{main_key}_{sub_key}"
                if new_key not in exclude_keys:  # Ensure it's expected
                    flat_dict[new_key] = value
        else:
            if main_key not in exclude_keys:  # Direct key-value pair
                flat_dict[main_key] = sub_dict

    return flat_dict


def analyze_pdf(pdf_path):
    """Extracts text from a PDF, splits into chunks, runs AI detection, and predicts classification."""
    
    # ✅ Step 1: Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("⚠️ No text extracted from file. Skipping analysis.")
        return

    # ✅ Step 2: Chunk text for better AI detection
    chunks = chunk_text(text)
    results = []

    for chunk in chunks:
        # ✅ Step 3: Extract linguistic features
        features = extract_features(chunk, None)  # Ensure extract_features handles raw text

        # ✅ Step 4: Compute DetectGPT & RoBERTa Scores
        detectgpt_score = compute_log_probability_openvino(chunk)
        roberta_results = detect_ai_openvino_roberta(chunk)

        if roberta_results is None:
            print("⚠️ Warning: RoBERTa detection failed, setting probability to 0")
            roberta_ai_prob = 0
        else:
            roberta_ai_prob = roberta_results.get("ai_prob", 0)  # Get AI probability, default to 0 if missing

        features["detectgpt_score"] = detectgpt_score
        features["roberta_ai_prob"] = roberta_ai_prob

        # ✅ Step 5: Flatten features for model compatibility
        flat_features = flatten_features(features)
        results.append(flat_features)

    # ✅ Step 6: Compute average scores for the document
    avg_detectgpt = np.mean([res["detectgpt_score"] for res in results])
    avg_roberta = np.mean([res["roberta_ai_prob"] for res in results])

    # ✅ Step 7: Prepare for classification
    df_features = pd.DataFrame([{
        **results[0],  # Use first feature set
        "detectgpt_score": avg_detectgpt,
        "roberta_ai_prob": avg_roberta
    }])

    # ✅ Step 8: Ensure feature names match model training
    expected_features = model.feature_names_in_
    missing_features = [col for col in expected_features if col not in df_features.columns]

    if missing_features:
        print(f"⚠️ Missing features in input data: {missing_features}")
        for col in missing_features:
            df_features[col] = 0  # Fill missing features with default value

    # ✅ Step 9: Display DataFrame (Show All Columns)
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", 1000)  # Expand display width
    print("\n🔍 Extracted Features :")
    print(df_features.head())

    # ✅ Step 10: Predict AI vs. Human
    try:
        prediction = model.predict(df_features)[0]
    except Exception as e:
        print(f"❌ Error in Prediction: {e}")
        return

    # ✅ Step 11: Print results
    print("\n🔍 AI Detection Results:")
    print(f"🎯 **Avg DetectGPT Score:** {avg_detectgpt:.4f}")
    print(f"🎯 **Avg RoBERTa AI Probability:** {avg_roberta:.4f}")
    print(f"🚀 **Final Verdict:** {'AI-Generated' if prediction == 1 else 'Human-Written'}")


# ✅ Run AI detection on a specific PDF
pdf_path = r"C:\Users\Ernie\Documents\GitHub\AI_Detector\documents\book1.pdf"
analyze_pdf(pdf_path)