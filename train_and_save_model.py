import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ✅ Load the undersampled dataset
file_path = "scraped_text/dataset_undersampled.csv"  # Ensure this file exists
df = pd.read_csv(file_path)

# ✅ Drop rows with NaN values
df.dropna(inplace=True)

# ✅ Separate features & labels
X = df.drop(columns=["label"])
y = df["label"]

# ✅ Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ✅ Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save the trained model
MODEL_PATH = "random_forest_ai_detector.pkl"
joblib.dump(model, MODEL_PATH)

print(f"✅ Model trained and saved as `{MODEL_PATH}`")
