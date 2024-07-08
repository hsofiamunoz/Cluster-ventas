import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Asumimos que los datos están en un DataFrame de pandas llamado df

# Eliminar columnas que terminen con 'Nuevok'
df = df.loc[:, ~df.columns.str.endswith('Nuevok')]

# Opción de selección
opciones = ["Análisis de cluster", "Análisis con transacciones inusuales"]
sel = 1
result = sel  # Simulamos la selección del usuario

if result == 1:
    variable = "cluster"
elif result == 2:
    df = df[df['ncluster'] == "C_inusuales"]
    Nuevok = sorted(df['cluster'].unique())
    nkluster = ", ".join(Nuevok)
    # Aquí se esperaría la selección del usuario basada en nkluster

# Selección del número de variables a graficar
nvars = 2  # Simulamos la selección del usuario
listseries = df.columns.tolist()

# Gráfico de las variables seleccionadas
variables_seleccionadas = listseries[:nvars]
if nvars == 2:
    sns.scatterplot(data=df, x=variables_seleccionadas[0], y=variables_seleccionadas[1], hue=variable)
elif nvars == 3:
    sns.pairplot(df[variables_seleccionadas + [variable]], hue=variable)
# Agregar más condiciones según el número de variables

plt.show()

# Análisis de componentes principales (PCA)
variables_explicativas = ["precio", "unidades", "venta_fact", "costo", "margen", "margen_contrib"]
X = df[variables_explicativas].dropna()

# Estandarización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicación de PCA
pca = PCA()
pca_result = pca.fit_transform(X_scaled)

# DataFrame con resultados de PCA
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

# Gráfico de biplot de PCA
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'])
plt.title('Análisis de Componentes Principales')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()

# Adición de variables z ortogonales al DataFrame original
df[['z1', 'z2', 'z3']] = pca_df[['PC1', 'PC2', 'PC3']]

# Cálculo del índice compuesto
df['Indice'] = df['z1'] * pca.explained_variance_ratio_[0] + df['z2'] * pca.explained_variance_ratio_[1] + df['z3'] * pca.explained_variance_ratio_[2]

# Ordenar DataFrame por el índice compuesto
df = df.sort_values('Indice', ascending=False)

# Filtrar y graficar los mejores vendedores
top_n = 10
best_sellers = df.head(top_n)

plt.figure(figsize=(12, 6))
sns.barplot(x='material', y='Indice', data=best_sellers)
plt.title(f'Top {top_n} Best Sellers')
plt.show()

# División en cuadrantes
df['Cuadrante'] = np.where((df['prueba1'] > 0) & (df['prueba2'] < 0), 'A',
                           np.where((df['prueba1'] > 0) & (df['prueba2'] > 0), 'B',
                                    np.where((df['prueba1'] < 0) & (df['prueba2'] > 0), 'C', 'D')))

# Mostrar resumen de estadísticas
stats = df.describe().T
print(stats)    