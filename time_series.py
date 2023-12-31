import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Örnek veri oluşturma
np.random.seed(42)
time_steps = 10000
data = np.sin(np.arange(0, 20 * np.pi, 20 * np.pi / time_steps)) + np.random.normal(0, 0.1, time_steps)

# Veriyi DataFrame'e dönüştürme
df = pd.DataFrame(data, columns=["Value"])

# Veriyi normalize etme
scaler = MinMaxScaler()
df["Value"] = scaler.fit_transform(df[["Value"]])

# Zaman serisini özellik ve etiket olarak ayırma
X = df["Value"].values[:-1]
y = df["Value"].values[1:]

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Veriyi model giriş şekline dönüştürme
X_train = X_train.reshape(-1, 1, 1)
X_test = X_test.reshape(-1, 1, 1)

# Model oluşturma
model = Sequential()
model.add(GRU(units=50, activation='tanh', input_shape=(1, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Gelecekteki 100 değeri tahmin etmek için
future_steps = 500

# Model girişini oluşturma
last_value = X_test[-1:]
future_input = []

for _ in range(future_steps):
    last_value = model.predict(last_value.reshape(-1, 1, 1))
    future_input.append(last_value[0, 0])  # Düzeltme burada

# Gelecekteki değerleri normalize etme
future_input = np.array(future_input).reshape(-1, 1, 1)
future_predictions = model.predict(future_input)

# Tahminleri gerçek değerlerle birleştirme
all_predictions = np.concatenate([y_test.reshape(-1, 1), scaler.inverse_transform(future_predictions.reshape(-1, 1))])

print(f"y_test: {all_predictions}")
# Gerçek ve tahmin edilen değerleri grafikleme
# Tahminleri gerçek değerlerle karşılaştırma
plt.plot(y_test, label='Gerçek Değerler')
plt.plot(all_predictions, label='Gelecek Tahminler')
plt.grid(True)
plt.legend()
plt.show()

#print(f"Future 100 points {future_predictions}")