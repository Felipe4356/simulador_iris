from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Cargar datos
iris = load_iris()
X = iris.data
y = iris.target

# 2. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# 4. Guardar modelo
os.makedirs("model", exist_ok=True)
joblib.dump(modelo, "model/modelo.pkl")
print("✅ Modelo guardado en model/modelo.pkl")
