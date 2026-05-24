import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("data/matches.csv")

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

# =====================================
# SELECT REQUIRED COLUMNS
# =====================================

df = df[
    [
        'team1',
        'team2',
        'toss_winner',
        'toss_decision',
        'venue',
        'match_winner'
    ]
]

# =====================================
# REMOVE NULL VALUES
# =====================================

df.dropna(inplace=True)

# =====================================
# REMOVE NO RESULT
# =====================================

df = df[df['match_winner'] != 'No Result']

# =====================================
# RESET INDEX
# =====================================

df.reset_index(drop=True, inplace=True)

# =====================================
# ENCODE FEATURES
# =====================================

encoders = {}

feature_cols = [
    'team1',
    'team2',
    'toss_winner',
    'toss_decision',
    'venue'
]

for col in feature_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

# =====================================
# ENCODE TARGET
# =====================================

target_encoder = LabelEncoder()

df['match_winner'] = target_encoder.fit_transform(
    df['match_winner']
)

encoders['match_winner'] = target_encoder

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop('match_winner', axis=1)

y = df['match_winner']

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# FIX LABELS
# =====================================

unique_classes = sorted(y_train.unique())

class_mapping = {
    old: new
    for new, old in enumerate(unique_classes)
}

reverse_mapping = {
    v: k
    for k, v in class_mapping.items()
}

y_train_fixed = y_train.map(class_mapping)

# =====================================
# MODEL
# =====================================

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='mlogloss'
)

# =====================================
# TRAIN MODEL
# =====================================

print("\nTraining Model...\n")

model.fit(X_train, y_train_fixed)

# =====================================
# PREDICTIONS
# =====================================

predictions = model.predict(X_test)

# Convert back to original labels
predictions_original = [
    reverse_mapping[p]
    for p in predictions
]

# =====================================
# ACCURACY
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions_original
)

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")

# =====================================
# SAVE MODEL
# =====================================

pickle.dump(
    model,
    open("model/xgb_model.pkl", "wb")
)

pickle.dump(
    encoders,
    open("model/encoders.pkl", "wb")
)

pickle.dump(
    class_mapping,
    open("model/class_mapping.pkl", "wb")
)

print("\n✅ Model Saved Successfully")
print("✅ Encoders Saved Successfully")

# =====================================
# SAMPLE PREDICTION
# =====================================

sample = X_test.iloc[0:1]

pred = model.predict(sample)[0]

original_pred = reverse_mapping[pred]

winner = encoders['match_winner'].inverse_transform(
    [original_pred]
)[0]

print(f"\n🏆 Sample Predicted Winner: {winner}")