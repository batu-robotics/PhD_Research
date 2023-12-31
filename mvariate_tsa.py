import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import GRU, Dense

# Örnek veri oluşturma
np.random.seed(42)
time_steps = 10000

# Beş değişkenli zaman serisi oluşturma
data = np.zeros((time_steps, 5))
for i in range(5):
    data[:,i] = 10*i*np.sin(np.arange(0, 20 * np.pi, 20 * np.pi / time_steps)) + np.random.normal(0, 0.1, time_steps)

# Veriyi DataFrame'e dönüştürme
df = pd.DataFrame(data, columns=["Value1", "Value2", "Value3", "Value4", "Value5"])

# Veriyi normalize etme
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Zaman serisini özellik ve etiket olarak ayırma
X = df_scaled.iloc[:-1, :]
y = df_scaled.iloc[1:, :]

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Veriyi model giriş şekline dönüştürme
X_train = np.array(X_train).reshape(-1, 1, 5)
X_test = np.array(X_test).reshape(-1, 1, 5)

# Model oluşturma
model = Sequential()
model.add(GRU(units=50, activation='tanh', return_sequences=True, input_shape=(1, 5)))
model.add(GRU(units=100, activation='tanh', return_sequences=True))
model.add(GRU(units=100, activation='tanh', return_sequences=True))
model.add(GRU(units=50, activation='tanh', return_sequences=True))
model.add(Dense(units=5))  # Çıkış boyutu 5
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, epochs=100, batch_size=64)

# Gelecekteki 100 değeri tahmin etmek için
future_steps = 500
last_value = X_test[-1:]

future_input = []
for _ in range(future_steps):
    last_value = model.predict(last_value.reshape(-1, 1, 5))
    future_input.append(last_value[0, 0])

# Gelecekteki değerleri normalize etme
future_input = np.array(future_input).reshape(-1, 1, 5)
future_predictions = model.predict(future_input)

# Tahminleri gerçek değerlerle birleştirme
all_predictions = np.concatenate([y_test.values, scaler.inverse_transform(future_predictions.reshape(-1, 5))])

# Gerçek ve tahmin edilen değerleri grafikleme
plt.plot(all_predictions)
plt.legend(df.columns)
plt.show()
