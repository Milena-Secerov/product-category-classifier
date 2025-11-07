import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# -----------------------
# 1️⃣ Путање
# -----------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "products.csv")
model_path = os.path.join(base_dir, "..", "models", "product_cat_model.pkl")

# -----------------------
# 2️⃣ Учитавање података
# -----------------------
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()  # уклања размака у заглављима
print("🔹 Учитавање података готово")
print(df.head())

# -----------------------
# 3️⃣ Чишћење података
# -----------------------
# Уклонити редове где је Product Title или Category Label NaN
df = df.dropna(subset=['Product Title', 'Category Label'])


# Такође може да провери колоне
print(f"Broj redova posle uklanjanja NaN: {len(df)}")

# -----------------------
# 4️⃣ Feature Engineering
# -----------------------
X = df['Product Title']
y = df['Category Label']

# Разделимо на тренинг и тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Претварање текста у TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# -----------------------
# 5️⃣ Тренирање модела
# -----------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vect, y_train)

# -----------------------
# 6️⃣ Evaluacija
# -----------------------
y_pred = model.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------
# 7️⃣ Снимање модела
# -----------------------
with open(model_path, 'wb') as f:
    # Сачувамо tuple: (model, vectorizer)
    pickle.dump((model, vectorizer), f)

print(f"✅ Модел је сачуван у {model_path}")

