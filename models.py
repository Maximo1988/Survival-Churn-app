import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
import os
import argparse
import uuid

# Flags for plot control: use --show-plots to display, --save-plots to save plots to ./plots
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--show-plots', action='store_true', help='Display plots interactively')
parser.add_argument('--save-plots', action='store_true', help='Save plots to an output directory')
parser.add_argument('--outdir', default='plots', help='Output directory for saved plots')
_parsed_args, _ = parser.parse_known_args()
SHOW_PLOTS = bool(_parsed_args.show_plots)
SAVE_PLOTS = bool(_parsed_args.save_plots)
OUTDIR = os.path.normpath(str(_parsed_args.outdir))

def finalize_plot(name=None):
    fig = plt.gcf()
    if SAVE_PLOTS:
        os.makedirs(OUTDIR, exist_ok=True)
        fname = f"{name or 'plot'}_{uuid.uuid4().hex[:8]}.png"
        outpath = os.path.join(OUTDIR, fname)
        fig.savefig(outpath, bbox_inches='tight')
        # Informar dónde se guardó la figura (útil en scripts no interactivos)
        if SHOW_PLOTS:
            print(f"Saved plot: {outpath}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

script_dir = os.path.dirname(__file__)
url = os.path.join(script_dir, "Telco_customer_churn.csv")
df = pd.read_csv(url)

if __name__ == "__main__":
    print("Working dir:", os.getcwd())
    print("Data shape:", df.shape)
    print(df.head(10))
    print()
    df.info()

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
ax.set_title("Distribución de clientes por Tenure", fontsize=14)
ax.set_xlabel("Intervalo Tenure (Meses)")
ax.set_ylabel("Número de clientes")
ax.tick_params(axis='x', rotation=90) # Rotar etiquetas del eje x

# Porcentajes sobre cada barra
for p, pct in zip(ax.patches, tabla_hist["% sobre total"]):
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{pct:.1f}%',
                    (p.get_x() + p.get_width()/2., height + ax.get_ylim()[1] * 0.02), # Aumento del offset vertical
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
plt.subplots_adjust(right=0.75) # Ajusta el margen derecho para dar espacio a la tabla
finalize_plot('tenure_hist')

conteo = df[(df['Tenure Months'] == 1) & (df['Churn Value'] == 1)].shape[0]
#clientes que han estado por 1 mes y de ellos, filtramos a los que se han ido.
print(conteo)

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
finalize_plot('churn_by_tenure')
plt.close()

# Total de clientes
total_clientes = len(df)

# Construir los mismos bins que usaste en el histograma
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
finalize_plot('churn_0_72')

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
finalize_plot('churn_acumulado')

try:
    from IPython.display import Math, display
    _has_ipython = True
except Exception:
    _has_ipython = False

formula = r"""
\hat{S}(t) = \prod_{t_i \leq t} \left( 1 - \frac{d_i}{n_i} \right)
"""
if _has_ipython:
    display(Math(formula))

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

finalize_plot('km_curve')

##LOG-RANK
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
finalize_plot('logrank_pvalues')

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
finalize_plot('km_by_categorical')

#Verificar valores faltantes.
#La unica columna que tiene valores NaN es 'Churn Reason'
df.isna().sum();

churn_reason_nan = df['Churn Reason'].isna().sum()

total_registros= df.shape[0]

porcen_churn_reason_nan =churn_reason_nan  /total_registros
porcen_churn_reason_nan

df.shape
#total de clientes vs total de variables (7,043 VS 33)

# creamos una copia del DF original
df_cox = df.copy()

# Columnas clave para el modelo de Cox
dur_col = 'Tenure Months'   # duración
evt_col = 'Churn Value'     # evento (1=churn, 0=censura)

# Descartamos las columnas identificadoras y de ubicación que no aportan al riesgo
drop_id_loc = [
    'CustomerID', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
    'Country', 'State', 'City'
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

#One-Hot Encoding:

categorical_features = ['Gender','Senior Citizen','Partner','Dependents',
                        'Phone Service','Contract','Internet Service',
                        'Paperless Billing','Payment Method']

# Codificamos las variables a sustitutos numericos y convertimos nuevas cols a binarias
df_dummies = pd.get_dummies(df[categorical_features], drop_first=True)

# Se combinan las columnas numéricas y codificadas y creamos un nuevo df
df_cox = df[['Tenure Months', 'Churn Value', 'Monthly Charges', 'Total Charges']].copy()
df_cox = pd.concat([df_cox, df_dummies], axis=1) #columnas
df_cox

from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(X):
    # Convertir todo a numérico
    X = X.apply(pd.to_numeric, errors='coerce')
    # Reemplazar infinitos por NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    # Eliminar columnas completamente vacías
    X = X.dropna(axis=1, how='all')
    # Eliminar filas con NaN
    X = X.dropna()

    vif_data = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    })
    return vif_data.sort_values(by="VIF", ascending=False)

# Excluir columnas problemáticas conocidas
pre_excluir = ['Monthly Charges', 'Total Charges']

# Construir lista de covariables (todas menos duración, evento y las excluidas)
covars_all = [c for c in df_cox.columns if c not in [dur_col, evt_col] + pre_excluir]

# Subset de covariables
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

#AJUSTE DE MODELO DE COX (COMPLETO):
from lifelines import CoxPHFitter

cph = CoxPHFitter(penalizer=0.01)  # penalización ligera para estabilidad
cph.fit(df_final, duration_col=dur_col, event_col=evt_col)
cph.print_summary()

df_final.columns

#AJUSTE DE MODELO DE COX (CON HR)
from lifelines import CoxPHFitter

from lifelines.statistics import proportional_hazard_test

# Ajustar modelo Cox
cph = CoxPHFitter()
cph.fit(df_final, duration_col=dur_col, event_col=evt_col)

# Test de riesgos proporcionales
results = proportional_hazard_test(cph, df_final, time_transform='rank')

# DataFrame con p-values
ph_test = results.summary

# Clasificar variables
violating = ph_test.loc[ph_test['p'] < 0.05].index.tolist()
not_violating = ph_test.loc[ph_test['p'] >= 0.05].index.tolist()

print("Variables que VIOLAN el supuesto de Proporcionalidad:")
for v in violating:
    print(f" - {v} (p={ph_test.loc[v,'p']:.4f})")

print("Variables que NO lo violan:")
for v in not_violating:
    print(f" - {v} (p={ph_test.loc[v,'p']:.4f})")
    
#AJUSTE DE MODELO DE ESTRATIFICACION: 

#Dividiremos la muestra en subgrupos

from lifelines import CoxPHFitter

# Definir listas
violating_vars = [
    'Contract_One year',
    'Contract_Two year',
    'Dependents_Yes',
    'Internet Service_Fiber optic',
    'Partner_Yes',
    'Payment Method_Electronic check',
    'Payment Method_Mailed check'
]

non_violating_vars = [
    'Gender_Male',
    'Internet Service_No',
    'Paperless Billing_Yes',
    'Payment Method_Credit card (automatic)',
    'Phone Service_Yes',
    'Senior Citizen_Yes'
]

# Ajustamos el modelo con estratificación
cph = CoxPHFitter()
cph.fit(
    df_final,
    duration_col=dur_col,
    event_col=evt_col,
    strata=violating_vars
)

cph.print_summary()

#AJUSTE MODELO DE COX CON INTERACCIONES TEMPORALES (CPH):

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

#COMPARACION DE 3 MODELOS COX:
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
import numpy as np

# Hold-out
train, test = train_test_split(
    df_final, test_size=0.3, random_state=42, stratify=df_final[evt_col]
)

# ---------------------------
# 1. Modelo completo
cph_full = CoxPHFitter(penalizer=0.01)
cph_full.fit(train, duration_col=dur_col, event_col=evt_col)

cindex_train_full = concordance_index(train[dur_col], -cph_full.predict_partial_hazard(train), train[evt_col])
cindex_test_full  = concordance_index(test[dur_col],  -cph_full.predict_partial_hazard(test),  test[evt_col])

# ---------------------------
# 2. Modelo estratificado
violating_vars = [
    'Contract_One year','Contract_Two year','Dependents_Yes',
    'Internet Service_Fiber optic','Partner_Yes',
    'Payment Method_Electronic check','Payment Method_Mailed check'
]

cph_strata = CoxPHFitter(penalizer=0.01)
cph_strata.fit(train, duration_col=dur_col, event_col=evt_col, strata=violating_vars)

cindex_train_strata = concordance_index(train[dur_col], -cph_strata.predict_partial_hazard(train), train[evt_col])
cindex_test_strata  = concordance_index(test[dur_col],  -cph_strata.predict_partial_hazard(test),  test[evt_col])

# ---------------------------
# 3. Modelo dependiente del tiempo (contratos * log(t))
train = train.copy()
test  = test.copy()

train['log_t'] = np.log(train[dur_col].clip(lower=0.1))
test['log_t']  = np.log(test[dur_col].clip(lower=0.1))

train['Contract_One_year_logt'] = train['Contract_One year'] * train['log_t']
test['Contract_One_year_logt']  = test['Contract_One year'] * test['log_t']

train['Contract_Two_year_logt'] = train['Contract_Two year'] * train['log_t']
test['Contract_Two_year_logt']  = test['Contract_Two year'] * test['log_t']

cph_td = CoxPHFitter(penalizer=0.01)
cph_td.fit(train, duration_col=dur_col, event_col=evt_col)

cindex_train_td = concordance_index(train[dur_col], -cph_td.predict_partial_hazard(train), train[evt_col])
cindex_test_td  = concordance_index(test[dur_col],  -cph_td.predict_partial_hazard(test),  test[evt_col])

print(f"Completo = Train: {cindex_train_full:.3f}, Test: {cindex_test_full:.3f}")
print(f"Estratificado = Train: {cindex_train_strata:.3f}, Test: {cindex_test_strata:.3f}")
print(f"Dependiente = Train: {cindex_train_td:.3f}, Test: {cindex_test_td:.3f}")

#PREDICCIONES
#DEFINIENDO UN CLIENTE NUEVO:

# Obtenemos las columnas del modelo que contienen los nombres de las covariables
columnas_modelo = cph.params_.index

# Creamos un DataFrame con todas las columnas inicializadas en 0
nuevo_cliente = pd.DataFrame({col: [0] for col in columnas_modelo})

# Asignamos valores hipotéticos:

# Contrato menor a 1 año (contrato de 1 año activo)
nuevo_cliente['Contract_One year'] = 1
nuevo_cliente['Contract_Two year'] = 0

# Variables adicionales
nuevo_cliente['Phone Service_Yes'] = 1
nuevo_cliente['Internet Service_Fiber optic'] = 1
nuevo_cliente['Paperless Billing_Yes'] = 1
nuevo_cliente['Partner_Yes'] = 0
nuevo_cliente['Payment Method_Electronic check'] = 1
nuevo_cliente['Internet Service_No'] = 0
nuevo_cliente['Gender_Male'] = 1
nuevo_cliente['Payment Method_Mailed check'] = 0
nuevo_cliente['Payment Method_Credit card (automatic)'] = 0
nuevo_cliente['Dependents_Yes'] = 0
nuevo_cliente['Senior Citizen_Yes'] = 1

# Obtener curva estimada (hasta mes 72)
surv_nuevo = cph.predict_survival_function(nuevo_cliente).loc[:72]

# Graficamos
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(surv_nuevo.index, surv_nuevo.values.squeeze(),
        linewidth=3, color='green', label='Cox (Cliente nuevo)')

ax.set_ylabel("Probabilidad de supervivencia")
ax.set_xlabel("Tiempo en meses")
ax.set_title("Curva de supervivencia estimada — Cliente nuevo")
ax.set_xlim(left=0)
ax.set_ylim(0.0, 1.0)
ax.legend()
ax.grid(True)

meses = [1, 2, 5, 12, 24, 48, 72]

for mes in meses:
    if mes in surv_nuevo.index:
        prob = float(surv_nuevo.loc[mes].values.squeeze())
        ax.plot(mes, prob, 'o', color='red')
        ax.vlines(x=mes, ymin=0, ymax=prob, color='gray', linestyle='--', linewidth=1)
        ax.hlines(y=prob, xmin=0, xmax=mes, color='gray', linestyle='--', linewidth=1)
        ax.text(mes, prob + 0.02, f"({mes}, {prob*100:.2f}%)",
                fontsize=9, color='blue', ha='center')

finalize_plot('cox_new_cliente')

#DEFINIENDO UN CLIENTE DE ALTO RIESGO:

import pandas as pd
import matplotlib.pyplot as plt

# Creamos un DataFrame con todas las columnas inicializadas en 0
columnas_modelo = cph.params_.index
cliente_alto = pd.DataFrame({col: [0] for col in columnas_modelo})

# Asignamos los valores de alto riesgo
cliente_alto['Contract_Month-to-month'] = 1
cliente_alto['Contract_One year'] = 0
cliente_alto['Contract_Two year'] = 0
cliente_alto['Internet Service_Fiber optic'] = 1
cliente_alto['Payment Method_Electronic check'] = 1
cliente_alto['Paperless Billing_Yes'] = 1
cliente_alto['Dependents_Yes'] = 0
cliente_alto['Internet Service_No'] = 0
cliente_alto['Phone Service_Yes'] = 1
cliente_alto['Partner_Yes'] = 0
cliente_alto['Gender_Male'] = 1
cliente_alto['Senior Citizen_Yes'] = 1
cliente_alto['Payment Method_Mailed check'] = 0
cliente_alto['Payment Method_Credit card (automatic)'] = 0

# Curva de supervivencia estimada (hasta mes 72)
surv_alto = cph.predict_survival_function(cliente_alto).loc[:72]

# Graficamos
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(surv_alto.index, surv_alto.values.squeeze(),
        linewidth=3, color='red', label='Cox (Cliente alto riesgo)')

ax.set_ylabel("Probabilidad de supervivencia")
ax.set_xlabel("Tiempo en meses")
ax.set_title("Curva de supervivencia estimada — Cliente alto riesgo")
ax.set_xlim(left=0)
ax.set_ylim(0.0, 1.0)
ax.legend()
ax.grid(True)

meses = [1, 2, 5, 12, 24, 48, 72]

for mes in meses:
    if mes in surv_alto.index:
        prob = float(surv_alto.loc[mes].values.squeeze())
        ax.plot(mes, prob, 'o', color='blue')
        ax.vlines(x=mes, ymin=0, ymax=prob, color='gray', linestyle='--', linewidth=1)
        ax.hlines(y=prob, xmin=0, xmax=mes, color='gray', linestyle='--', linewidth=1)
        ax.text(mes, prob + 0.02, f"({mes}, {prob*100:.2f}%)",
                fontsize=9, color='blue', ha='center')

finalize_plot('cox_cliente_alto')

