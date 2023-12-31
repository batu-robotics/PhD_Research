import pandas as pd
import matplotlib.pyplot as plt

# Örnek bir veri seti oluşturulması
veri = {'Sayisal_Degerler': [5, 8, 3, 10, 7, 6, 2, 9, 4],
        'Binary_Degerler': [1, 0, 1, 0, 1, 0, 1, 0, 1]}

df = pd.DataFrame(veri)

# 'Binary_Degerler' sütununa göre gruplandırma
gruplar = df.groupby('Binary_Degerler')['Sayisal_Degerler']

# Her bir gruptaki verilerle histogram oluşturma
fig, ax = plt.subplots()
for label, grup in gruplar:
    grup.plot(kind="line", alpha=0.5, label=f'Binary_Degerler={label}', ax=ax)

# Grafik özelliklerinin ayarlanması
ax.legend()
ax.set_title('Sayisal Degerlerin Histogrami Gruplandirma ile')
ax.set_xlabel('Sayisal Degerler')
ax.set_ylabel('Frekans')

# Grafiği göster
plt.show()