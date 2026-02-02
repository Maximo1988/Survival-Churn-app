import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import itertools
import math
from IPython.display import Math, display

import os

# Cargar datos desde archivo local
SCRIPT_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(SCRIPT_DIR, "Telco_customer_churn.csv")
df = pd.read_csv(CSV_PATH)
df.head(5)

df.shape # (7043 clientes, 33 columnas)
df.columns
# Index(['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
       #'Lat Long', 'Latitude', 'Longitude', 'Gender', 'Senior Citizen',
       #'Partner', 'Dependents', 'Tenure Months', 'Phone Service',
       #'Multiple Lines', 'Internet Service', 'Online Security',
       #'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
       #'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method',
       #'Monthly Charges', 'Total Charges', 'Churn Label', 'Churn Value',
       #'Churn Score', 'CLTV', 'Churn Reason'],
       # dtype='object')

df['Churn Value'].value_counts()
#Esta variable binaria seria la de interes, donde:
# 1 = churn (abandono)
# 0 = activo

df['Tenure Months'].value_counts().head()
#meses vs la permanencia de clientes

# Total de clientes
total_clientes = len(df)

# Histograma con 30 bins
conteos, bins = np.histogram(df['Tenure Months'], bins=30)
porcentajes = (conteos / total_clientes * 100).round(2)

# Crear tabla con intervalos, conteos y porcentajes
tabla_hist = pd.DataFrame({
    "Intervalo Tenure (Meses)": [f"{int(bins[i])}–{int(bins[i+1])}" for i in range(len(bins)-1)],
    "Clientes": conteos,
    "% sobre total": porcentajes
})

# Filtrar primeros 4 intervalos y últimos 3 para el resumen
tabla_primeros4 = tabla_hist.iloc[:4]
tabla_ultimos3 = tabla_hist.iloc[-3:]
tabla_resumen = pd.concat([tabla_primeros4, tabla_ultimos3])

# --- Crear la figura y el subgráfico ---
fig, ax = plt.subplots(figsize=(16, 7))

sns.barplot(x=tabla_hist["Intervalo Tenure (Meses)"], y=tabla_hist["Clientes"],
            color="skyblue", edgecolor="black", ax=ax)

# Resaltar primeros 4 intervalos en naranja
for i, bar in enumerate(ax.patches[:4]):
    bar.set_facecolor("orange")

# Resaltar últimos 3 intervalos en verde
for i, bar in enumerate(ax.patches[-3:]):
    bar.set_facecolor("green")

# Título y etiquetas para el gráfico de barras
ax.set_title("Distribución de clientes por Tenure)", fontsize=14)
ax.set_xlabel("Intervalo Tenure (Meses)")
ax.set_ylabel("Número de clientes")
ax.tick_params(axis='x', rotation=90) # Rotar etiquetas del eje x

# Porcentajes sobre cada barra
for p, pct in zip(ax.patches, tabla_hist["% sobre total"]):
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{pct:.1f}%',
                    (p.get_x() + p.get_width()/2., height + ax.get_ylim()[1] * 0.02), # Aumentar el desplazamiento vertical
                    ha='center', va='bottom', fontsize=7, color="black", rotation=90, clip_on=False)

# Crear la tabla usando los datos resumidos dentro del gráfico
# Ajustar bbox para que quepa bien la columna y no tape los porcentajes
table = ax.table(cellText=tabla_resumen.values,
                       colLabels=tabla_resumen.columns,
                       cellLoc='center',
                       loc='upper right',
                       bbox=[0.70, 0.65, 0.3, 0.3]) # Ajustado: x más a la izquierda, ancho un poco mayor

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(0.7, 0.7) # Escalar la tabla para mejor ajuste

# Ajustando el layout para hacer espacio a la tabla
plt.tight_layout()
plt.subplots_adjust(right=0.75) # Ajustar el margen derecho para dar espacio a la tabla
plt.show()

#Comentarios:
# - Un 12.2% de los clientes permanecen en la telefonica entre 0 y 2.
# - Un 5.3% permanecen entre 2 y 7 meses.
# - Entre 9 y 64 tienen tasas de permanencia que varian entre 1.7% y 4.3%.
# - Entre 69-72 meses la tasa de permanencia es del 9.24%.

conteo = df[(df['Tenure Months'] == 1) & (df['Churn Value'] == 1)].shape[0]
#clientes que han estado por 1 mes y de ellos, filtramos a los que se han ido.
print(conteo)
#380

df_churn = df[df['Churn Value'] == 1].copy()
#solo clientes que han abandonado el servicio

# Totales del estudio
total_clientes = len(df)                               # todos los que comenzaron
clientes_activos = df[df['Churn Value'] == 0].shape[0]  # los que quedaron activos
total_churn = len(df_churn)                            # los que se retiraron

# Conteos por mes
max_tenure = int(df['Tenure Months'].max())
conteos = (
    df_churn['Tenure Months']
    .astype(int)
    .value_counts()
    .reindex(range(0, max_tenure + 1), fill_value=0)
)

# Porcentajes por mes respecto al total churn
porcentajes = (conteos / total_churn * 100).round(1)

# Gráfico
plt.figure(figsize=(14,6))
ax = sns.barplot(x=conteos.index, y=conteos.values, color="red")

plt.title("Distribución de clientes que hicieron Churn por Tenure (por mes)")
plt.xlabel("Tenure (Meses)")
plt.ylabel("Número de clientes (Churn)")
plt.tight_layout()

# Posición central aproximada para las anotaciones
x_centro = max_tenure // 2
y_ref = conteos.max() * 0.7 if conteos.max() > 0 else 1

# Anotaciones centradas
plt.text(0.1 * max_tenure, y_ref,
         f"Comenzaron: {total_clientes}",
         fontsize=12, color="blue", ha="center")

# Texto hacia el marco derecho (Quedaron activos)
plt.text(0.9 * max_tenure, y_ref - (conteos.max()*0.15),
         f"Quedaron activos: {clientes_activos}",
         fontsize=12, color="green", ha="center")

# Porcentajes sobre cada barra
for x, (cnt, pct) in enumerate(zip(conteos.values, porcentajes.values)):
    if cnt > 0:
        ax.annotate(f'{pct:.1f}%',
                    (x, cnt + max(1, cnt * 0.03)),  # pequeño desplazamiento vertical
                    ha='center', va='bottom', fontsize=8, color="black", rotation=90)

# Ticks del eje X (cada 2 meses para legibilidad)
plt.xticks(range(0, max_tenure + 1, 2))
plt.show()
plt.close()

# Total de clientes
total_clientes = len(df)

# Construir histograma con los mismos bins anteriores
conteos, bins = np.histogram(df['Tenure Months'], bins=30)

# Calcular porcentajes
porcentajes = (conteos / total_clientes * 100).round(2)

# Crear tabla con intervalos, conteos y porcentajes
tabla_hist = pd.DataFrame({
    "Intervalo Tenure (Meses)": [f"{int(bins[i])}–{int(bins[i+1])}" for i in range(len(bins)-1)],
    "Clientes": conteos,
    "% sobre total": porcentajes
})

# Mostrar tabla completa
print(tabla_hist.to_string(index=False))

# Totales
total_clientes = len(df)              # 7032
clientes_activos = (df['Churn Value'] == 0).sum()
total_churn = total_clientes - clientes_activos  # 1869

# Conteos por mes (solo churn) y rango hasta 72
conteos = (
    df[df['Churn Value'] == 1]['Tenure Months']
    .astype(int)
    .value_counts()
    .reindex(range(0, 73), fill_value=0)
)

# Porcentajes respecto al total de clientes (7032)
porcentajes_total = (conteos / total_clientes * 100)

# Verificación
print(f"Suma porcentajes respecto al total: {porcentajes_total.sum():.2f}%")
print(f"Esto equivale a {(total_churn/total_clientes*100):.2f}% de churn global")

# Gráfico
plt.figure(figsize=(14,6))
ax = sns.barplot(x=conteos.index, y=conteos.values, color="red")

plt.title("Distribución de churn por Tenure (0–72 meses) respecto al total de clientes")
plt.xlabel("Tenure (Meses)")
plt.ylabel("Número de clientes (Churn)")
plt.tight_layout()

# Anotaciones ejecutivas
y_ref = conteos.max() * 0.7 if conteos.max() > 0 else 1
plt.text(0.1 * 72, y_ref, f"Comenzaron: {total_clientes}", fontsize=12, color="blue", ha="center")
plt.text(0.9 * 72, y_ref - (conteos.max()*0.15), f"Quedaron activos: {clientes_activos}", fontsize=12, color="green", ha="center")

# Porcentajes sobre cada barra (respecto al total de 7032)
for x, (cnt, pct) in enumerate(zip(conteos.values, porcentajes_total.values)):
    if cnt > 0:
        ax.annotate(f'{pct:.2f}%',
                    (x, cnt + max(1, cnt * 0.03)),
                    ha='center', va='bottom', fontsize=8, color="black", rotation=90)

plt.xticks(range(0, 73, 2))
plt.xlim(-0.5, 72.5)
plt.show()

# Suma porcentajes respecto al total: 26.54%
# Esto equivale a 26.54% de churn global

# Totales
total_clientes = len(df)              # 7032
clientes_activos = (df['Churn Value'] == 0).sum()
total_churn = total_clientes - clientes_activos  # 1869

# Conteos por mes (solo churn) y rango hasta 72
conteos = (
    df[df['Churn Value'] == 1]['Tenure Months']
    .astype(int)
    .value_counts()
    .reindex(range(0, 73), fill_value=0)
)

# Acumulado de churn por mes
acumulado = conteos.cumsum()
porcentaje_acumulado = acumulado / total_clientes * 100

# Meses de interés elegidos:
meses = [1, 2, 5, 12, 24, 48, 72]

#PORQUE?
# Meses 1, 2 y 5: Periodo critico de la relacion del cliente, asi como el abandono temprano.
# Meses 12 y 24: Fin de contratos de 1 y 2 años
# Meses 48 y 72: Lealtad del cliente

# Gráfico acumulado
plt.figure(figsize=(14,6))
plt.plot(porcentaje_acumulado.index, porcentaje_acumulado.values,
         marker='o', color='red', linewidth=2, label="Churn acumulado (%)")

# Puntos destacados
for mes in meses:
    pct = porcentaje_acumulado[mes]
    plt.plot(mes, pct, 'o', color='blue')
    # Líneas guía hasta el punto
    plt.vlines(x=mes, ymin=0, ymax=pct, color='gray', linestyle='--', linewidth=1)
    plt.hlines(y=pct, xmin=0, xmax=mes, color='gray', linestyle='--', linewidth=1)
    # Texto con coordenadas por encima del punto
    plt.text(mes, pct + 0.5, f"({mes}, {pct:.2f}%)",
             fontsize=9, color='blue', ha='center')

# Configuración del gráfico
plt.title("Churn acumulado por Tenure (0–72 meses) respecto al total de clientes")
plt.xlabel("Tenure (Meses)")
plt.ylabel("Churn acumulado (%)")
plt.xticks(range(0, 73, 6))
plt.xlim(0, 72)
plt.ylim(0, porcentaje_acumulado.max()*1.1)
plt.grid(alpha=0.3)
plt.legend()
plt.show()

#Comentarios:
# - Un 5.4% del total de clientes (7,032) se retiran el primer mes, mes en el cual se registra el mayor numero de retiros, lo cual representa un 20.3% de los 1,869 que abandonaron la telefonica.
# - Entre el 2do y 5to mes hay retiradas entre 0.91% y 1,75% con respecto al total.

#Metodologia y Modelo:
# ¿Por qué elegimos este modelo sobre otros, como Regresión Logística, Random Forest, Gradient Boosting o Redes Neuronales?
#1. Por el manejo de la censura de datos, el cual nos ayuda a no sesgar los resultados como lo harían las Redes Neuronales.
#2. Porque predice no solo si un cliente abandonará el servicio, sino cuándo lo hará, al contrario de una Regresión Logística.
#3. Nos ayuda a entender cómo cambia el riesgo de abandono a lo largo del tiempo, sin clasificar de manera binaria (sí/no o 0/1); como lo haría un Random Forest.
#4. Para calcular la vida del cliente de forma más precisa, dato que no es bien definido utilizando Boosting.

#####################################
#Curva de supervivencia (Kaplan–Meier)
#Edward Kaplan y Paul Meier presentaron el estimador Kaplan–Meier.
# Técnica no paramétrica para estimar la probabilidad de supervivencia en función del tiempo, especialmente útil cuando existen datos incompletos o censurados.
# Este método se convirtió en la herramienta estándar para analizar y visualizar la duración hasta un evento en medicina, ingeniería y negocios.

#Formulacion Matematica:
#Sean:
#S(t): el estimador no parametrico de la supervivencia, mide la probabilidad de que un individuo sobreviva más allá de un tiempo t.
#t_1<t_2<\dots <t_k: los tiempos en los que ocurren eventos (abandono, falla, muerte).
# d_i: número de eventos ocurridos en el tiempo t_i.
# n_i: número de individuos en riesgo justo antes de t_i.
#[ \hat{S}(t) = \prod_{t_i \leq t} \left( 1 - \frac{d_i}{n_i} \right) ]

from IPython.display import Math, display
formula = r"""
\hat{S}(t) = \prod_{t_i \leq t} \left( 1 - \frac{d_i}{n_i} \right)
"""
display(Math(formula))
#######################################

##1. Curva de Supervivencia KAPLAN-MEIER

# Ajustar el modelo
kmf = KaplanMeierFitter()
kmf.fit(df['Tenure Months'], df['Churn Value'], label='Supervivencia')

# Graficar la curva de supervivencia
ax = kmf.plot(linewidth=3, figsize=(10,6))
ax.set_ylabel("Probabilidad de supervivencia")
ax.set_xlabel("Tiempo en meses")
ax.set_title("Curva de supervivencia de clientes en la Telefónica")

# Forzar que los ejes comiencen en 0
ax.set_xlim(left=0)
ax.set_ylim(0.5, 1.0)   # curva más baja y centrada

# Meses de interés
meses = [1, 2, 5, 12, 24, 48, 72]

# Obtener valores directamente del modelo y dibujar líneas guía
for mes in meses:
    prob = kmf.predict(mes)  # probabilidad en ese mes

    # Punto en la curva
    ax.plot(mes, prob, 'o', color='red')

    # Líneas guía desde eje X y Y hasta el punto (sin sobrepasar la curva)
    ax.vlines(x=mes, ymin=0, ymax=prob, color='gray', linestyle='--', linewidth=1)
    ax.hlines(y=prob, xmin=0, xmax=mes, color='gray', linestyle='--', linewidth=1)

    # Texto con coordenadas por encima del punto
    ax.text(mes, prob + 0.02, f"({mes}, {prob*100:.2f}%)",
            fontsize=9, color='blue', ha='center')

plt.show()
#########################################

# Log Rank Test

#Verificamos cuales son las variables categoricas
df.info()

# Filtrar columnas de tipo object
object_cols = df.select_dtypes(include="object").columns

# Visualizar cuales son las categorias que componen cada variable categorica
for col in object_cols:
    print(f"\nColumna: {col}")
    print(df[col].unique())

# Log Rank sobre cada variable categorica:

logrank_p_values = {}

categorical_cols = ['Gender',
    'Senior Citizen',
    'Partner',
    'Dependents',
    'Phone Service',
    'Multiple Lines',
    'Internet Service',
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Contract',
    'Paperless Billing',
    'Payment Method',
]

for col in categorical_cols:
    unique_categories = df[col].unique()

    if len(unique_categories) == 2:
    # Prueba para variables binarias:
    #(Gender,Senior Citizen,Partner,Dependents,Phone Service,Paperless Billing)

        # Asignamos valores unicos
        cat1 = unique_categories[0]
        cat2 = unique_categories[1]

        # Creamos booleanos True/False
        df_cat1 = df[df[col] == cat1]
        df_cat2 = df[df[col] == cat2]
        results = logrank_test(durations_A=df_cat1['Tenure Months'],
                              event_observed_A=df_cat1['Churn Value'],
                              durations_B=df_cat2['Tenure Months'],
                              event_observed_B=df_cat2['Churn Value'])
        logrank_p_values[col] = results.p_value

    elif len(unique_categories) > 2:
    #pruebas log-rank por pares

        min_p_value_for_col = 1.0 #no pongo 0.05 porque si hay valores mas altos no saldrian

        # Recorrer todos los pares únicos de categorías para la columna actual
        for i in range(len(unique_categories)):
            for j in range(i + 1, len(unique_categories)):
                cat1 = unique_categories[i]
                cat2 = unique_categories[j]

                df_cat1 = df[df[col] == cat1]
                df_cat2 = df[df[col] == cat2]
                results = logrank_test(durations_A=df_cat1['Tenure Months'],
                                      event_observed_A=df_cat1['Churn Value'],
                                      durations_B=df_cat2['Tenure Months'],
                                      event_observed_B=df_cat2['Churn Value'])
                min_p_value_for_col = min(min_p_value_for_col, results.p_value)
        # Almacenando el valor-p mínimo de las variables multicategoricas dentro del diccionario
        logrank_p_values[col] = min_p_value_for_col

for key, value in sorted(logrank_p_values.items(), key=lambda x: x[1]):
    print(f"{key}: {value}")

# Grafico del Ranking de los pvalue obtenidos

# Convertimos el diccionario (logrank_p_values) a un df para graficar
p_values_df = pd.DataFrame(list(logrank_p_values.items()), columns=['Categorical Feature', 'P-value'])

# Ordenamos el p-value de manera ascendente para mejor entendimiento
p_values_df = p_values_df.sort_values(by='P-value', ascending=True)

plt.figure(figsize=(12, 7))
sns.barplot(x='P-value', y='Categorical Feature', data=p_values_df)
plt.axvline(x=0.05, color='red', linestyle='--', label='Nivel de Significancia (alpha = 0.05)')
plt.title('Log-Rank Test (p-value < 0.05)')
plt.xlabel('P-value')
plt.ylabel('Categorical Feature')
plt.xscale('log')
plt.legend()
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.show()

# Aplicacion de Curvas de Supervivencia a las variables con pvalue < 0.05
# para ver cuales tienen mas incidencia en Churn

import itertools
import math

kmf = KaplanMeierFitter()

significant_vars = [
    'Contract', 'Online Security', 'Tech Support', 'Payment Method',
    'Online Backup', 'Device Protection', 'Partner', 'Dependents',
    'Internet Service', 'Streaming Movies', 'Streaming TV',
    'Paperless Billing', 'Senior Citizen', 'Multiple Lines'
]

# Determinar el tamaño de la cuadrícula de subplots
num_vars = len(significant_vars)
ncols = 3 # Número de columnas que deseas
nrows = math.ceil(num_vars / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
# Aplanar la matriz de axes para facilitar la iteración si nrows > 1
axes = axes.flatten()

for i, variable in enumerate(significant_vars):
    ax = axes[i] # Seleccionar el subplot actual

    categories = df[variable].unique()
    results_text = []

    # Comparaciones por pares para Log-Rank Test
    for cat1, cat2 in itertools.combinations(categories, 2):
        mask1 = df[variable] == cat1
        mask2 = df[variable] == cat2

        result = logrank_test(
            df[mask1]['Tenure Months'], df[mask2]['Tenure Months'],
            event_observed_A=df[mask1]['Churn Value'],
            event_observed_B=df[mask2]['Churn Value']
        )
        pval = result.p_value
        results_text.append(f"{cat1} vs {cat2}: p={pval:.4f}")

    # Graficar curvas de supervivencia en el subplot actual
    for cat in categories:
        mask = df[variable] == cat
        kmf.fit(df[mask]['Tenure Months'], df[mask]['Churn Value'], label=str(cat))
        kmf.plot_survival_function(ax=ax)

    ax.set_title(f"Curvas de supervivencia por {variable}")
    ax.set_xlabel("Meses de Tenencia", fontsize=8)
    ax.set_ylabel("Probabilidad de Retención", fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(title=variable, loc='lower left', fontsize=7, title_fontsize=8)

    # Añadir los p-values como texto dentro del gráfico
    text_str = "\n".join(results_text)
    # Ajustar posición del texto en función del número de líneas para evitar solapamiento
    y_pos = 0.95 - (len(results_text) * 0.05) # Ajusta este valor si el texto se solapa
    ax.text(
        0.02, y_pos, text_str, transform=ax.transAxes, fontsize=7,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
    )

# Ocultar subplots vacíos si los hay
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.98]) # Ajustar layout para dejar espacio para el título general
fig.suptitle('Curvas de Supervivencia por Variables Categóricas Significativas', fontsize=16, y=1.0)
plt.show()

#Verificar valores faltantes.
#La unica columna que tiene valores NaN es 'Churn Reason'
df.isna().sum()

churn_reason_nan = df['Churn Reason'].isna().sum()

total_registros= df.shape[0]

porcen_churn_reason_nan =churn_reason_nan  /total_registros
porcen_churn_reason_nan

###########################################

# REGRESION COX:

df.shape
#total de clientes vs total de variables (7,043 VS 33)

# creamos una copia del DF original
df_cox = df.copy()

# Columnas clave para el modelo de Cox
dur_col = 'Tenure Months'   # duración
evt_col = 'Churn Value'     # evento (1=churn, 0=censura)

# Descartamos las columnas identificadoras y de ubicación que no aportan al riesgo
drop_id_loc = [
    'CustomerID', 'Count', 'Country', 'State', 'City',
    'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
    'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason',
    'Monthly Charges', 'Total Charges'
]


# Verificación de existencia de columnas clave
assert dur_col in df_cox.columns, f"Falta columna de duración: {dur_col}"
assert evt_col in df_cox.columns, f"Falta columna de evento: {evt_col}"

# Limpieza de nombres y tipos
df_cox.columns = df_cox.columns.str.strip()
df_cox[dur_col] = pd.to_numeric(df_cox[dur_col], errors='coerce')
df_cox[evt_col] = pd.to_numeric(df_cox[evt_col], errors='coerce')

# Filtrar filas válidas: duración > 0 y evento en {0,1}
df_cox = df_cox[(df_cox[dur_col] > 0) & (df_cox[evt_col].isin([0, 1]))]

# Eliminar columnas no informativas
df_cox = df_cox.drop(columns=[c for c in drop_id_loc if c in df_cox.columns])

print(df_cox.head(3))

# Codificacion de variables categoricas:
# En virtud que las columnas (Gender, Partner, Internet Service, Contract, etc.) son categoricas,
# es necesario convertirlas en dummies (variables binarias) para que el modelo de COX las use.

# One-Hot Encoding:
# Transformamos variables categóricas en variables binarias (0/1)

categorical_features = ['Gender','Senior Citizen','Partner','Dependents',
                        'Phone Service','Contract','Internet Service',
                        'Paperless Billing','Payment Method']

# Validar que todas las columnas categóricas existen
missing_cat = [col for col in categorical_features if col not in df.columns]
if missing_cat:
    print(f"Advertencia: Columnas categóricas faltantes: {missing_cat}")
    categorical_features = [col for col in categorical_features if col in df.columns]

# Codificamos las variables a sustitutos numericos y convertimos nuevas cols a binarias
df_dummies = pd.get_dummies(df[categorical_features], drop_first=True)

# Se combinan las columnas numéricas y codificadas y creamos un nuevo df
df_cox = df[['Tenure Months', 'Churn Value', 'Monthly Charges', 'Total Charges']].copy()
df_cox = pd.concat([df_cox, df_dummies], axis=1) #columnas
df_cox.head(3)

#Revisando como quedaron codificadas las variables
# Recorremos todas las variables categóricas originales
for col in categorical_features:
    # Buscar las columnas dummy que corresponden a esa variable
    related = [c for c in df_cox.columns if c.startswith(col+"_")]

    if related:  # si existen columnas dummy para esa variable
        print(f"\nVariable original: {col}")
        print("Categorías originales:", df[col].unique())
        print("Columnas dummy creadas:", related)
        # Agrupamos por la categoría original y mostramos la media de las dummies
        print(df_cox.groupby(df[col])[related].mean())

# Analisis de Multicolinealidad usando la tecnica VIF (Variance Inflation Factor):

from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(X):
    """Calcular VIF para detectar multicolinealidad"""
    try:
        # Convertir todo a numérico
        X = X.apply(pd.to_numeric, errors='coerce')
        # Reemplazar infinitos por NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        # Eliminar columnas completamente vacías
        X = X.dropna(axis=1, how='all')
        # Eliminar filas con NaN
        X = X.dropna()
        
        if X.shape[1] == 0 or X.shape[0] == 0:
            raise ValueError("Dataset vacío después de limpieza")

        vif_data = pd.DataFrame({
            "feature": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        })
        return vif_data.sort_values(by="VIF", ascending=False)
    except Exception as e:
        print(f"Error en check_vif: {e}")
        return pd.DataFrame()

# Excluir columnas problemáticas conocidas
pre_excluir = ['Monthly Charges', 'Total Charges']

# Construir lista de covariables (todas menos duración, evento y las excluidas)
covars_all = [c for c in df_cox.columns if c not in [dur_col, evt_col] + pre_excluir]

# Subconjunto de covariables
X_vif = df_cox[covars_all].copy()

# Convertir booleanos a enteros (0/1) antes de calcular VIF
bool_cols = X_vif.select_dtypes(include=['bool']).columns
X_vif[bool_cols] = X_vif[bool_cols].astype(int)

# Calcular VIF
vif_res = check_vif(X_vif)
print(vif_res.head(10))

# Filtrar por umbral VIF
umbral_vif = 10
covars_final = vif_res.loc[vif_res['VIF'] <= umbral_vif, 'feature'].tolist()

# Reconstruir dataset final para Cox
df_final = pd.concat([df_cox[[dur_col, evt_col]], df_cox[covars_final]], axis=1)
print("Shape final:", df_final.shape)

print(X_vif.dtypes)
#Confirmacion de que las columnas booleanas fueron correctamente convertidas a enteros
#Esta conversion y validacion nos ayuda a poder hacer el VIF con certeza

#########################################

# Ajuste del Modelo COX (completo):
try:
    cph = CoxPHFitter(penalizer=0.01)  # penalización ligera para estabilidad
    cph.fit(df_final, duration_col=dur_col, event_col=evt_col)
    cph.print_summary()
    print("Modelo COX ajustado exitosamente")
except Exception as e:
    print(f"Error al ajustar modelo COX: {e}")
    cph = None

# Ajustar modelo Cox con manejo de errores
try:
    cph = CoxPHFitter()
    cph.fit(df_final, duration_col=dur_col, event_col=evt_col)
    
    # Test de riesgos proporcionales
    results = proportional_hazard_test(cph, df_final, time_transform='rank')
    ph_test = results.summary
    
    # Clasificar variables
    violating = ph_test.loc[ph_test['p'] < 0.05].index.tolist()
    not_violating = ph_test.loc[ph_test['p'] >= 0.05].index.tolist()
    
    print("Variables que VIOLAN el supuesto de Proporcionalidad:")
    for v in violating:
        print(f" - {v} (p={ph_test.loc[v,'p']:.4f})")
    
    print("\nVariables que NO lo violan:")
    for v in not_violating:
        print(f" - {v} (p={ph_test.loc[v,'p']:.4f})")
except Exception as e:
    print(f"Error en Test de Proporcionalidad: {e}")
    violating = []
    not_violating = []

# NOTA:
# Si alguna variable viola el supuesto de la Hipotesis de Proporcionalidad:
#1. Estratificamos si es categórica.
#2. Agregamos interacciones con tiempo si es continua.

########################################

# Modelo de Estratificación con manejo de errores
try:
    # Definir listas de variables
    violating_vars = [
        'Contract_One year',
        'Contract_Two year',
        'Dependents_Yes',
        'Internet Service_Fiber optic',
        'Partner_Yes',
        'Payment Method_Electronic check',
        'Payment Method_Mailed check'
    ]
    
    # Filtrar solo las variables que existen en df_final
    violating_vars = [v for v in violating_vars if v in df_final.columns]
    
    if violating_vars:
        cph_strat = CoxPHFitter()
        cph_strat.fit(
            df_final,
            duration_col=dur_col,
            event_col=evt_col,
            strata=violating_vars
        )
        cph_strat.print_summary()
        print("Modelo COX con estratificación ajustado exitosamente")
    else:
        print("Advertencia: No hay variables de estratificación disponibles")
except Exception as e:
    print(f"Error en modelo estratificado: {e}")

# Tecnica de Interaccion con el tiempo:


# --- Parámetros ---
dur_col = 'Tenure Months'
evt_col = 'Churn Value'

time_interaction_vars = ['Contract_One year', 'Contract_Two year']

strata_candidates = [
    'Phone Service_Yes',
    'Internet Service_No',
    'Paperless Billing_Yes',
    'Payment Method_Credit card (automatic)',
    'Gender_Male',
    'Senior Citizen_Yes',
    'Dependents_Yes',
    'Partner_Yes',
    'Payment Method_Electronic check',
    'Payment Method_Mailed check'
]

# --- 0) Validaciones básicas ---
assert dur_col in df_final.columns, f"{dur_col} no está en df_final"
assert evt_col in df_final.columns, f"{evt_col} no está en df_final"

# --- 1) Copia y normalización de etiqueta ---
df_all = df_final.copy()
if df_all[evt_col].dtype == object or not set(df_all[evt_col].unique()).issubset({0,1}):
    mapping = {'Yes':1,'No':0,'yes':1,'no':0,'Y':1,'N':0,'True':1,'False':0}
    df_all[evt_col] = df_all[evt_col].map(mapping).fillna(df_all[evt_col]).astype(int)

# --- 2) Filtrar duraciones inválidas (aplicar antes de crear log_t) ---
df_all = df_all[df_all[dur_col] > 0].copy()

# --- 3) Convertir categóricas a dummies (si existen) sobre todo el dataset ---
cat_cols = df_all.select_dtypes(include=['object','category']).columns.tolist()
if len(cat_cols) > 0:
    df_all = pd.get_dummies(df_all, columns=cat_cols, drop_first=True)

# --- 4) Asegurar columnas de strata candidatas (filtrar las que existen) ---
strata_cols = [c for c in strata_candidates if c in df_all.columns]

# --- 5) Forzar binariedad en strata (si es posible) y eliminar no binarias ---
valid_strata = []
for s in strata_cols:
    vals = set(df_all[s].dropna().unique())
    if vals.issubset({0,1}):
        valid_strata.append(s)
    else:
        # intentar convertir a 0/1 si es razonable
        try:
            df_all[s] = pd.to_numeric(df_all[s], errors='coerce').fillna(0).astype(int)
            if set(df_all[s].unique()).issubset({0,1}):
                valid_strata.append(s)
        except Exception:
            pass
strata_cols = valid_strata

# --- 6) Crear log_t sobre todo el dataset ---
df_all['log_t'] = np.log(df_all[dur_col].clip(lower=0.1))

# --- 7) Crear interacciones solo para las variables de contrato (si existen) ---
for c in time_interaction_vars:
    if c in df_all.columns:
        df_all[c] = pd.to_numeric(df_all[c], errors='coerce').fillna(0).astype(int)
        inter_col = f"{c.replace(' ', '_')}_logt"
        df_all[inter_col] = df_all[c] * df_all['log_t']

# --- 8) Lista final de features (excluye duración, evento y mantiene strata_cols) ---
features_all = [col for col in df_all.columns if col not in [dur_col, evt_col]]

# --- 9) Inspección rápida de la base final antes del split ---
print("Columnas en df_all (muestra):", features_all[:40])
print("Número de filas en df_all:", len(df_all))
print("Strata detectadas y usadas:", strata_cols)

# --- 10) Ahora sí: Hold-out split (estratificando por la etiqueta) ---
train, test = train_test_split(df_all, test_size=0.3, random_state=42, stratify=df_all[evt_col])

# 11) Asegurar que ambos conjuntos tengan las mismas columnas (alinear)
test = test.reindex(columns=train.columns, fill_value=0)

# 12) Construir df_for_fit y df_for_test con las columnas que usaremos en fit
#     (mantener dur_col, evt_col, todas las features y las strata_cols)
cols_for_model = [dur_col, evt_col] + [c for c in train.columns if c not in [dur_col, evt_col]]
df_for_fit  = train[cols_for_model].copy().dropna()
df_for_test = test[cols_for_model].copy().dropna()

print(f"Filas train antes dropna: {len(train)}, después: {len(df_for_fit)}")
print(f"Filas test  antes dropna: {len(test)},  después: {len(df_for_test)}")

# --- 13) Ajustar CoxPH estratificado en train (usando strata_cols detectadas) ---
cph = CoxPHFitter(penalizer=0.01)
cph.fit(df_for_fit, duration_col=dur_col, event_col=evt_col, strata=strata_cols)
cph.print_summary()

# --- 14) Concordance (train y test) con 4 decimales ---
pred_train = cph.predict_partial_hazard(df_for_fit)
c_index_train = concordance_index(df_for_fit[dur_col], -pred_train.values.ravel(), df_for_fit[evt_col])
print("C-index (train):", f"{c_index_train:.4f}")

pred_test = cph.predict_partial_hazard(df_for_test)
c_index_test = concordance_index(df_for_test[dur_col], -pred_test.values.ravel(), df_for_test[evt_col])
print("C-index (test):", f"{c_index_test:.4f}")

# --- 15) Métricas adicionales: partial log-likelihood, partial AIC, llr test ---
pll = float(cph.log_likelihood_)
k = cph.params_.shape[0]
partial_aic = -2.0 * pll + 2.0 * k
llr = cph.log_likelihood_ratio_test()
print(f"partial log-likelihood\t{pll:.2f}")
print(f"Partial AIC\t{partial_aic:.2f}")
print(f"log-likelihood ratio test\t{llr.test_statistic:.2f} on {llr.degrees_freedom} df")
print(f"p-value (llr)\t{llr.p_value:.4g}")

# --- 16) Proportional hazard test sobre covariables no-strata ---
cols_model_raw = list(cph.params_.index)
cols_for_ph_test = [c for c in cols_model_raw if c in df_for_fit.columns and c not in strata_cols]
if len(cols_for_ph_test) == 0:
    print("No hay covariables válidas para proportional_hazard_test (posible causa: todas son strata).")
else:
    ph_test = proportional_hazard_test(cph, df_for_fit, time_transform='rank', columns=cols_for_ph_test)
    print("\nproportional_hazard_test summary:")
    print(ph_test.summary)

df_all.shape
# (7032 clientes vs 18 variables)

##################################

# #AJUSTE MODELO DE COX CON INTERACCIONES TEMPORALES (CPH):

import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

# Hold-out split
train, test = train_test_split(
    df_final, test_size=0.3, random_state=42, stratify=df_final[evt_col]
)

# 1) Reemplazar duraciones <= 0 por un mínimo positivo (ej. 0.1)
train = train[train[dur_col] > 0].copy()
test  = test[test[dur_col] > 0].copy()

# 2) Crear columna log(t) sin NaNs
train['log_t'] = np.log(train[dur_col].clip(lower=0.1))
test['log_t']  = np.log(test[dur_col].clip(lower=0.1))

# 3) Creamos interacciones para las variables continuas
train['Contract_One_year_logt'] = train['Contract_One year'] * train['log_t']
test['Contract_One_year_logt']  = test['Contract_One year'] * test['log_t']

train['Contract_Two_year_logt'] = train['Contract_Two year'] * train['log_t']
test['Contract_Two_year_logt']  = test['Contract_Two year'] * test['log_t']

# 4) Eliminamos cualquier fila con NaN residual
train = train.dropna()
test  = test.dropna()

# 5) Entrenamos el modelo Cox
cph_td = CoxPHFitter(penalizer=0.01)
cph_td.fit(train, duration_col=dur_col, event_col=evt_col)

cph_td.print_summary()

#################################

# PREDICCIONES:

#1. Definiendo un cliente Bueno:



# -------------------------
# Helpers (copia las tuyas si ya existen)
# -------------------------
def build_feature_map_from_model_cols(model_cols):
    fmap = []
    for col in model_cols:
        col = str(col)
        if col == 'log_t' or col.endswith('_logt'):
            fmap.append((col, None))
            continue
        if ('_' in col or ' ' in col) and not col.replace('_','').replace(' ','').replace('.','').isdigit():
            if '_' in col:
                parts = col.rsplit('_', 1)
            else:
                parts = col.rsplit(' ', 1)
            feat, cat = parts[0], parts[1]
            fmap.append((feat, cat))
        else:
            fmap.append((col, None))
    return fmap

def profile_to_vector_from_model(profile_dict, feature_map):
    x = []
    for feat, cat in feature_map:
        if cat is None:
            x.append(float(profile_dict.get(feat, 0.0)))
        else:
            key_exact = f"{feat}_{cat}"
            if key_exact in profile_dict:
                x.append(float(profile_dict.get(key_exact, 0.0)))
            else:
                val = profile_dict.get(feat)
                if val is None:
                    x.append(0.0)
                else:
                    if isinstance(val, (int, float)):
                        x.append(1.0 if float(val) == 1.0 else 0.0)
                    else:
                        x.append(1.0 if str(val) == str(cat) else 0.0)
    return np.array(x, dtype=float)

def interp_baseline_H0_for_model(cph, t):
    H0_df = cph.baseline_cumulative_hazard_.copy()
    if isinstance(H0_df, pd.Series):
        s = H0_df.astype(float).copy()
    else:
        if H0_df.shape[1] == 1:
            s = H0_df.iloc[:,0].astype(float).copy()
        else:
            s = H0_df.iloc[:,0].astype(float).copy()
    times = s.index.values.astype(float)
    vals = s.values.astype(float).squeeze()
    return float(np.interp(t, times, vals))

# -------------------------
# Perfil bueno (usa el dict definido arriba)
# -------------------------
profile = {
    'Tenure Months': 6.0,
    'Contract_Month-to-month': 1,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'Internet Service_Fiber optic': 1,
    'Internet Service_No': 0,
    'Paperless Billing_Yes': 1,
    'Payment Method_Electronic check': 1,
    'Payment Method_Mailed check': 0,
    'Payment Method_Credit card (automatic)': 0,
    'Dependents_Yes': 0,
    'Partner_Yes': 0,
    'Phone Service_Yes': 1,
    'Gender_Male': 0,
    'Senior Citizen_Yes': 1
}

# -------------------------
# Construir vector X y factor exponencial
# -------------------------
model_cols = list(cph_td.params_.index)
fmap = build_feature_map_from_model_cols(model_cols)
X = profile_to_vector_from_model(profile, fmap)

# rellenar log_t si existe
if 'log_t' in model_cols:
    tenure = float(profile.get('Tenure Months', profile.get('tenure', 0.1)))
    X[model_cols.index('log_t')] = math.log(max(tenure, 0.1))

# manejar columnas *_logt
for i, col in enumerate(model_cols):
    if str(col).endswith('_logt'):
        base_name = str(col)[:-5]
        if X[i] == 0.0:
            key_alt = base_name
            key_alt2 = base_name.replace(' ', '_')
            base_val = float(profile.get(key_alt, profile.get(key_alt2, 0.0)))
            if 'log_t' in model_cols:
                logt_val = X[model_cols.index('log_t')]
            else:
                logt_val = math.log(max(float(profile.get('Tenure Months', 0.1)), 0.1))
            X[i] = base_val * logt_val

beta = cph_td.params_.values.astype(float).squeeze()
if X.shape[0] != beta.shape[0]:
    raise ValueError(f"Dim mismatch: X={X.shape[0]} vs beta={beta.shape[0]}")

exp_factor = float(np.exp(float(np.dot(beta, X))))
print("exp(beta·X) =", exp_factor)

# -------------------------
# Calcular S(t) en 0..72
# -------------------------
t_grid = np.linspace(0, 72, 721)
Lambda0_grid = np.array([interp_baseline_H0_for_model(cph_td, t) for t in t_grid])
Lambda_grid = Lambda0_grid * exp_factor
S_grid = np.exp(-Lambda_grid)
surv_df = pd.DataFrame({'time': t_grid, 'S': S_grid})

# imprimir S en meses clave
for m in [1,5,12,24,48,72]:
    s_m = float(np.interp(m, t_grid, S_grid))
    print(f"S({m}) = {s_m:.6g}")

# -------------------------
# Graficar (curva naranja, marcadores en 5,12,24,48)
# -------------------------
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(surv_df['time'], surv_df['S'], color='green', linewidth=3, label='S(t) perfil bueno')

# marcadores y etiquetas en meses clave
for m in [5,12,24,48]:
    s_m = float(np.interp(m, t_grid, S_grid))
    ax.plot(m, s_m, 'o', color='#ff7f0e', markersize=9, markeredgecolor='white', markeredgewidth=1.2)
    ax.text(m, s_m - 0.04, f"{m}m: {s_m*100:.2f}%", color='#ff7f0e', fontsize=9, ha='center', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

ax.set_xlim(0,72)
ax.set_ylim(0.0,1.02)
ax.set_xlabel("Tiempo (meses)")
ax.set_ylabel("Probabilidad de supervivencia S(t)")
ax.set_title("Curva de supervivencia estimada — Perfil Bueno (propuesto)")
ax.grid(False)
ax.legend()
plt.show()

# 2. Definiendo un Cliente de riesgo malo:

# Versión modificada: la curva y anotaciones indican explícitamente "Perfil malo"
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# -------------------------
# Helpers (idénticos a los usados antes)
# -------------------------
def build_feature_map_from_model_cols(model_cols):
    fmap = []
    for col in model_cols:
        col = str(col)
        if col == 'log_t' or col.endswith('_logt'):
            fmap.append((col, None))
            continue
        if ('_' in col or ' ' in col) and not col.replace('_','').replace(' ','').replace('.','').isdigit():
            if '_' in col:
                parts = col.rsplit('_', 1)
            else:
                parts = col.rsplit(' ', 1)
            feat, cat = parts[0], parts[1]
            fmap.append((feat, cat))
        else:
            fmap.append((col, None))
    return fmap

def profile_to_vector_from_model(profile_dict, feature_map):
    x = []
    for feat, cat in feature_map:
        if cat is None:
            x.append(float(profile_dict.get(feat, 0.0)))
        else:
            val = profile_dict.get(feat)
            if val is None:
                key_exact = f"{feat}_{cat}"
                x.append(float(profile_dict.get(key_exact, 0.0)))
            else:
                if isinstance(val, (int, float)):
                    x.append(1.0 if float(val) == 1.0 else 0.0)
                else:
                    x.append(1.0 if str(val) == str(cat) else 0.0)
    return np.array(x, dtype=float)

def interp_baseline_H0_for_model(cph, t):
    H0_df = cph.baseline_cumulative_hazard_.copy()
    if isinstance(H0_df, pd.Series):
        s = H0_df.astype(float).copy()
    else:
        if H0_df.shape[1] == 1:
            s = H0_df.iloc[:,0].astype(float).copy()
        else:
            col = H0_df.columns[0]
            s = H0_df[col].astype(float).copy()
    times = s.index.values.astype(float)
    vals = s.values.astype(float).squeeze()
    return float(np.interp(t, times, vals))

# -------------------------
# Perfil (cliente regular)
# -------------------------
profile_dict = {
    'Tenure Months': 1.0,
    'Contract_Month-to-month': 1,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'Internet Service_Fiber optic': 1,
    'Internet Service_No': 0,
    'Paperless Billing_Yes': 1,
    'Payment Method_Electronic check': 1,
    'Payment Method_Mailed check': 0,
    'Payment Method_Credit card (automatic)': 0,
    'Dependents_Yes': 0,
    'Partner_Yes': 0,
    'Phone Service_Yes': 0,
    'Gender_Male': 0,
    'Senior Citizen_Yes': 0
}

# -------------------------
# Construir S(t) para el perfil en 0..72
# -------------------------
model_cols = list(cph_td.params_.index)
fmap = build_feature_map_from_model_cols(model_cols)
X = profile_to_vector_from_model(profile_dict, fmap)

# rellenar log_t si existe (usar Tenure Months)
if 'log_t' in model_cols:
    tenure = float(profile_dict.get('Tenure Months', profile_dict.get('tenure', 0.1)))
    logt_val = math.log(max(tenure, 0.1))
    X[model_cols.index('log_t')] = logt_val

# manejar columnas *_logt
for i, col in enumerate(model_cols):
    if str(col).endswith('_logt'):
        base_name = str(col)[:-5]
        if X[i] == 0.0:
            key_alt = base_name
            key_alt2 = base_name.replace(' ', '_')
            base_val = float(profile_dict.get(key_alt, profile_dict.get(key_alt2, 0.0)))
            if 'log_t' in model_cols:
                logt_val = X[model_cols.index('log_t')]
            else:
                logt_val = math.log(max(float(profile_dict.get('Tenure Months', 0.1)), 0.1))
            X[i] = base_val * logt_val

beta = cph_td.params_.values.astype(float).squeeze()
if X.shape[0] != beta.shape[0]:
    raise ValueError(f"Dimensión mismatch: X={X.shape[0]} vs beta={beta.shape[0]}")

exp_factor = float(np.exp(float(np.dot(beta, X))))

# grilla de tiempos y cálculo de S(t)
t_grid = np.linspace(0, 72, 721)  # paso 0.1 mes
Lambda0_grid = np.array([interp_baseline_H0_for_model(cph_td, t) for t in t_grid])
Lambda_grid = Lambda0_grid * exp_factor
S_grid = np.exp(-Lambda_grid)

# DataFrame con la supervivencia del perfil
surv_profile_df = pd.DataFrame({'time': t_grid, 'S': S_grid})

# -------------------------
# Plot: curva en regulary anotación explícita "Perfil: malo"
# -------------------------
y_min, y_max = 0.2, 1.0
x_min, x_max = 0, 72

fig, ax = plt.subplots(figsize=(10,6))

# Curva (perfil regular) en Malo
ax.plot(surv_profile_df['time'], surv_profile_df['S'],
        linewidth=3, color='green', label='Cox (Perfil Malo)', zorder=2)

# Puntos de interés y marcadores en 5, 12 y 24 meses (rojo con borde blanco)
for m in [5, 12, 24]:
    s_m = float(np.interp(m, surv_profile_df['time'].values, surv_profile_df['S'].values))
    ax.plot(m, s_m, marker='o', color='red', markersize=9, markeredgecolor='white', markeredgewidth=1.2, zorder=3)
    ax.text(m, s_m - 0.03, f"{m}m: {s_m*100:.2f}%", fontsize=9, color='red', ha='center', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=4)

# Añadir caja de texto que indica explícitamente que es un cliente malo
ax.text(0.98, 0.02, "Perfil: Malo", transform=ax.transAxes,
        fontsize=11, color='white', ha='right', va='bottom',
        bbox=dict(facecolor='red', alpha=0.9, edgecolor='none', pad=6), zorder=5)

# Ejes y título
ax.set_xlabel("Tiempo en meses")
ax.set_ylabel("Probabilidad de supervivencia")
ax.set_title("Curva de supervivencia estimada — Perfil de cliente (Malo)", fontsize=14)

# Límites solicitados (x termina en 72)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Sin cuadrícula
ax.grid(False)

# Ticks en x cada 6 meses
ax.set_xticks(np.arange(0, x_max + 1, 6))
ax.set_xticklabels([str(int(t)) for t in np.arange(0, x_max + 1, 6)])

# Puntos de interés y anotaciones con líneas de intersección (azules -> cambiados a rojo para consistencia)
meses = [1, 12, 24, 48, 72]
idx = surv_profile_df['time'].values.astype(float)
vals = surv_profile_df['S'].values.astype(float)

label_gap = 0.06 * (y_max - y_min)
offset_steps = [0.0, 1.0 * label_gap, 2.0 * label_gap]

for i, mes in enumerate(meses):
    prob = float(np.interp(mes, idx, vals))
    if mes < x_min or mes > x_max or prob < y_min:
        continue

    # dibujar punto (rojo)
    ax.plot(mes, prob, 'o', color='red', markersize=6, zorder=3)

    # líneas de intersección en gris
    ax.vlines(x=mes, ymin=y_min, ymax=prob, color='gray', linestyle='--', linewidth=1, zorder=1)
    ax.hlines(y=prob, xmin=x_min, xmax=mes, color='gray', linestyle='--', linewidth=1, zorder=1)

    extra = offset_steps[i] if i < 3 else 0.0
    y_label = prob - label_gap - extra
    if y_label < (y_min + 0.005):
        y_label = prob + label_gap

    ax.text(mes, y_label, f"({mes}, {prob*100:.2f}%)", fontsize=9, color='red',
            ha='center', va='top' if y_label <= prob else 'bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1), zorder=4)

ax.legend()
plt.show()

#################################

# Calculo de las probabilidades condicionales de Churn:

#La condicion para este tipo de probabilidad es que la persona haya estado en la telefonica al momento t1
# y que su Churn ocurra entre t1 y t2.
# Esa condicion se puede expresar matematicamente como:

#          P(t_1<T≤t_2∣T>t_1)


#  Definir funcion auxiliar
def churn_prob_interval_from_profile_cph(model, profile_dict, t1, t2):
    # lógica de probabilidad absoluta
    ...
    return prob

import math

# --- 1) Define aquí el perfil que quieras evaluar ---
# Ejemplo: perfil "malo" (alto riesgo)
profile_dict = {
    'Tenure Months': 1.0,
    'Contract_Month-to-month': 1,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'Internet Service_Fiber optic': 1,
    'Internet Service_No': 0,
    'Paperless Billing_Yes': 1,
    'Payment Method_Electronic check': 1,
    'Payment Method_Mailed check': 0,
    'Payment Method_Credit card (automatic)': 0,
    'Dependents_Yes': 0,
    'Partner_Yes': 0,
    'Phone Service_Yes': 0,
    'Gender_Male': 0,
    'Senior Citizen_Yes': 0
}

# --- 2) Intervalo de interés ---
t1, t2 = 1.0, 12.0   # meses

# --- 3) Probabilidad absoluta (P(t1 < T <= t2)) usando la función existente ---
prob_abs = churn_prob_interval_from_profile_cph(cph_td, profile_dict, t1, t2)

# --- 4) Calcular componentes para la probabilidad condicional ---
# Reconstruir X y exp_factor (usa las funciones auxiliares que definiste)
model_cols = list(cph_td.params_.index)
fmap = build_feature_map_from_model_cols(model_cols)
X = profile_to_vector_from_model(profile_dict, fmap)
beta = cph_td.params_.values.astype(float).squeeze()
exp_factor = float(math.exp(float(np.dot(beta, X))))

# Baseline cumulative hazard en t1 y t2
Lam1_0 = interp_baseline_H0_for_model(cph_td, t1)
Lam2_0 = interp_baseline_H0_for_model(cph_td, t2)

# Lambda y supervivencias
Lam1 = Lam1_0 * exp_factor
Lam2 = Lam2_0 * exp_factor
S1 = math.exp(-Lam1)
S2 = math.exp(-Lam2)

# Probabilidad condicional dado supervivencia a t1
prob_cond = 1.0 - math.exp(-(Lam2 - Lam1))
# alternativa numérica (debe coincidir salvo redondeos)
prob_cond_alt = prob_abs / S1 if S1 > 0 else float('nan')

# --- 5) Mostrar resultados ---
print("Perfil usado:", profile_dict)
print(f"P_abs (t1,t2] = {prob_abs:.6g}")
print(f"S(t1) = {S1:.6g}, S(t2) = {S2:.6g}")
print(f"exp(beta·X) = {exp_factor:.6g}")
print(f"Lambda0(t1) = {Lam1_0:.6g}, Lambda0(t2) = {Lam2_0:.6g}")
print(f"Lambda(t1) = {Lam1:.6g}, Lambda(t2) = {Lam2:.6g}")
print()

print(f"P_conditional | T>t1 = {prob_cond:.6g} ")

#####################################

import math

# --- 1) Define aquí el perfil que quieras evaluar ---
# Ejemplo: perfil "malo" (alto riesgo)
profile_dict = {
    'Tenure Months': 1.0,
    'Contract_Month-to-month': 1,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'Internet Service_Fiber optic': 1,
    'Internet Service_No': 0,
    'Paperless Billing_Yes': 1,
    'Payment Method_Electronic check': 1,
    'Payment Method_Mailed check': 0,
    'Payment Method_Credit card (automatic)': 0,
    'Dependents_Yes': 0,
    'Partner_Yes': 0,
    'Phone Service_Yes': 0,
    'Gender_Male': 0,
    'Senior Citizen_Yes': 0
}

# --- 2) Intervalo de interés ---
t1, t2 = 6.0, 12.0   # meses

# --- 3) Probabilidad absoluta (P(t1 < T <= t2)) usando la función existente ---
prob_abs = churn_prob_interval_from_profile_cph(cph_td, profile_dict, t1, t2)

# --- 4) Calcular componentes para la probabilidad condicional ---
# Reconstruir X y exp_factor (usa las funciones auxiliares que definiste)
model_cols = list(cph_td.params_.index)
fmap = build_feature_map_from_model_cols(model_cols)
X = profile_to_vector_from_model(profile_dict, fmap)
beta = cph_td.params_.values.astype(float).squeeze()
exp_factor = float(math.exp(float(np.dot(beta, X))))

# Baseline cumulative hazard en t1 y t2
Lam1_0 = interp_baseline_H0_for_model(cph_td, t1)
Lam2_0 = interp_baseline_H0_for_model(cph_td, t2)

# Lambda y supervivencias
Lam1 = Lam1_0 * exp_factor
Lam2 = Lam2_0 * exp_factor
S1 = math.exp(-Lam1)
S2 = math.exp(-Lam2)

# Probabilidad condicional dado supervivencia a t1
prob_cond = 1.0 - math.exp(-(Lam2 - Lam1))
# alternativa numérica (debe coincidir salvo redondeos)
prob_cond_alt = prob_abs / S1 if S1 > 0 else float('nan')

# --- 5) Mostrar resultados ---
print("Perfil usado:", profile_dict)
print(f"P_abs (t1,t2] = {prob_abs:.6g}")
print(f"S(t1) = {S1:.6g}, S(t2) = {S2:.6g}")
print(f"exp(beta·X) = {exp_factor:.6g}")
print(f"Lambda0(t1) = {Lam1_0:.6g}, Lambda0(t2) = {Lam2_0:.6g}")
print(f"Lambda(t1) = {Lam1:.6g}, Lambda(t2) = {Lam2:.6g}")
print()

print(f"P_conditional | T>t1 = {prob_cond:.6g} ")

import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1) Perfil de cliente a evaluar ---
profile_dict = {
    'Tenure Months': 1.0,
    'Contract_Month-to-month': 1,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'Internet Service_Fiber optic': 1,
    'Internet Service_No': 0,
    'Paperless Billing_Yes': 1,
    'Payment Method_Electronic check': 1,
    'Payment Method_Mailed check': 0,
    'Payment Method_Credit card (automatic)': 0,
    'Dependents_Yes': 0,
    'Partner_Yes': 0,
    'Phone Service_Yes': 0,
    'Gender_Male': 0,
    'Senior Citizen_Yes': 0
}

# --- 2) Función para calcular probabilidad condicional en un intervalo ---
def prob_conditional_interval(model, profile_dict, t1, t2):
    model_cols = list(model.params_.index)
    fmap = build_feature_map_from_model_cols(model_cols)
    X = profile_to_vector_from_model(profile_dict, fmap)
    beta = model.params_.values.astype(float).squeeze()
    exp_factor = float(np.exp(np.dot(beta, X)))

    Lam1_0 = interp_baseline_H0_for_model(model, t1)
    Lam2_0 = interp_baseline_H0_for_model(model, t2)

    Lam1 = Lam1_0 * exp_factor
    Lam2 = Lam2_0 * exp_factor

    # Probabilidad condicional
    prob_cond = 1.0 - math.exp(-(Lam2 - Lam1))
    return prob_cond

# --- 3) Construir el "reloj" ---
# Intervalos de tiempo (ejemplo: meses 1 a 12)
t_values = np.arange(1, 13)  # meses
probs = []

for t in t_values[:-1]:
    p = prob_conditional_interval(cph_td, profile_dict, t, t+1)
    probs.append(p)

# --- 4) Graficar ---
plt.figure(figsize=(10,6))
plt.plot(t_values[:-1], probs, marker='o', linestyle='-', color='navy')
plt.title("El Reloj del Churn: Probabilidad Condicional por Mes")
plt.xlabel("Mes")
plt.ylabel("Probabilidad condicional de churn")
plt.grid(True)

# Marcar zonas críticas (ejemplo: prob > 0.3)
for i, p in enumerate(probs):
    if p > 0.3:
        plt.text(t_values[i], p, "⚠", fontsize=14, color="red")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1) Definir dos perfiles ---
# Perfil "malo" (alto riesgo)
profile_malo = {
    'Tenure Months': 1.0,
    'Contract_Month-to-month': 1,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'Internet Service_Fiber optic': 1,
    'Internet Service_No': 0,
    'Paperless Billing_Yes': 1,
    'Payment Method_Electronic check': 1,
    'Payment Method_Mailed check': 0,
    'Payment Method_Credit card (automatic)': 0,
    'Dependents_Yes': 0,
    'Partner_Yes': 0,
    'Phone Service_Yes': 0,
    'Gender_Male': 0,
    'Senior Citizen_Yes': 0
}

# Perfil "bueno" (bajo riesgo)
profile_bueno = {
    'Tenure Months': 12.0,
    'Contract_Month-to-month': 0,
    'Contract_One year': 0,
    'Contract_Two year': 1,
    'Internet Service_Fiber optic': 0,
    'Internet Service_No': 0,
    'Paperless Billing_Yes': 0,
    'Payment Method_Electronic check': 0,
    'Payment Method_Mailed check': 0,
    'Payment Method_Credit card (automatic)': 1,
    'Dependents_Yes': 1,
    'Partner_Yes': 1,
    'Phone Service_Yes': 1,
    'Gender_Male': 1,
    'Senior Citizen_Yes': 0
}

# --- 2) Función para calcular probabilidad condicional ---
def prob_conditional_interval(model, profile_dict, t1, t2):
    model_cols = list(model.params_.index)
    fmap = build_feature_map_from_model_cols(model_cols)
    X = profile_to_vector_from_model(profile_dict, fmap)
    beta = model.params_.values.astype(float).squeeze()
    exp_factor = float(np.exp(np.dot(beta, X)))

    Lam1_0 = interp_baseline_H0_for_model(model, t1)
    Lam2_0 = interp_baseline_H0_for_model(model, t2)

    Lam1 = Lam1_0 * exp_factor
    Lam2 = Lam2_0 * exp_factor

    prob_cond = 1.0 - math.exp(-(Lam2 - Lam1))
    return prob_cond

# --- 3) Construir el reloj para ambos perfiles ---
t_values = np.arange(1, 11)  # meses 1 a 10
probs_malo = []
probs_bueno = []

for t in t_values[:-1]:
    probs_malo.append(prob_conditional_interval(cph_td, profile_malo, t, t+1))
    probs_bueno.append(prob_conditional_interval(cph_td, profile_bueno, t, t+1))

# --- 4) Graficar ---
plt.figure(figsize=(10,6))
plt.plot(t_values[:-1], probs_malo, marker='o', linestyle='-', color='red', label='Perfil alto riesgo')
plt.plot(t_values[:-1], probs_bueno, marker='o', linestyle='-', color='green', label='Perfil bajo riesgo')

plt.title("El Reloj del Churn: Comparación de Perfiles")
plt.xlabel("Intervalo (mes t → t+1)")
plt.ylabel("Probabilidad condicional de churn")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Supongamos que ya tienes la lista de probabilidades condicionales
# probs = [p1, p2, p3, ...] calculadas con tu función prob_conditional_interval

# Calcular incrementos porcentuales
increments = []
for i in range(1, len(probs)):
    if probs[i-1] > 0:
        inc = (probs[i] - probs[i-1]) / probs[i-1] * 100
    else:
        inc = np.nan  # evitar división por cero
    increments.append(inc)

# Graficar
plt.figure(figsize=(10,6))
plt.plot(range(2, len(probs)+1), increments, marker='o', linestyle='-', color='purple')
plt.title("Incremento porcentual de la probabilidad condicional de churn")
plt.xlabel("Intervalo (mes t → t+1)")
plt.ylabel("Incremento porcentual (%)")
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Supongamos que ya calculaste las probabilidades condicionales
# probs = [p1, p2, p3, ...] para cada mes
# Aquí pongo un ejemplo ficticio
probs = [0.0002, 0.002, 0.005, 0.01, 0.015, 0.03, 0.05, 0.07, 0.09, 0.12]

# Ángulos para cada mes (convertidos a radianes)
months = np.arange(1, len(probs)+1)
angles = np.linspace(0, 2*np.pi, len(probs), endpoint=False)

# Gráfico polar
plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, probs, marker='o', linestyle='-', color='navy')
ax.fill(angles, probs, alpha=0.3, color='skyblue')

# Etiquetas de meses alrededor del círculo
ax.set_xticks(angles)
ax.set_xticklabels([f"Mes {m}" for m in months])

ax.set_title("Reloj del Churn (Probabilidad Condicional)", va='bottom')
plt.show()
