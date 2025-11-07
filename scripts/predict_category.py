import os
import pickle

# -----------------------
# Путање
# -----------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "models", "product_cat_model.pkl")

# -----------------------
# Учитавање модела
# -----------------------
if not os.path.exists(model_path):
    print(f"⚠️ Модел није пронађен у {model_path}. Прво покрени train_model.py")
    exit()

with open(model_path, 'rb') as f:
    model, vectorizer = pickle.load(f)

# -----------------------
# Интерактивно предвиђање
# -----------------------
print("🟢 Модел учитан. Унеси назив производа (или 'exit' за излаз):")

while True:
    product_title = input("Назив производа: ")
    if product_title.lower() == 'exit':
        break
    pred_vect = vectorizer.transform([product_title])
    pred_category = model.predict(pred_vect)[0]
    print(f"Предвиђена категорија: {pred_category}\n")
