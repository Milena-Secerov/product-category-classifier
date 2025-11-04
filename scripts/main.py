import os
import pandas as pd
import pickle

# -----------------------
# 1️⃣ Путање до фајлова
# -----------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "products.csv")
model_path = os.path.join(base_dir, "..", "models", "product_cat_model.pkl")

# -----------------------
# 2️⃣ Учитавање података
# -----------------------
print("🔹 Учитавање података...")
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()  # уклања размака у заглављима
print(df.head())

# -----------------------
# 3️⃣ Провера модела
# -----------------------
if os.path.exists(model_path):
    print("✅ Модел пронађен – учитавам...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    print("⚠️ Модел није пронађен. Биће креиран касније.")
