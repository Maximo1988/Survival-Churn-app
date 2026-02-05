# Survival-Churn-Analysis

## ¿Por qué elegimos este modelo sobre otros como Regresión Logística, Random Forest, Gradient Boosting o Redes Neuronales?

#1. Por el manejo de la censura de datos, el cual nos ayuda a no sesgar los resultados como lo harían las Redes Neuronales.

#2. Porque predice no solo si un cliente abandonará el servicio, sino cuándo lo hará, al contrario de una Regresión Logística.

#3. Nos ayuda a entender cómo cambia el riesgo de abandono a lo largo del tiempo, sin clasificar de manera binaria (sí/no o 0/1); como lo haría un Random Forest.

#4. Para calcular la vida del cliente de forma más precisa, dato que no es bien definido utilizando Boosting.


--------------------------

Breve guía de uso (CLI):
- Ejecuta el análisis principal (sin mostrar ni guardar figuras):

```powershell
python .\Survival-Churn-app\models.py
```

- Guardar figuras en la carpeta por defecto `plots/`:

```powershell
python .\Survival-Churn-app\models.py --save-plots
```

- Mostrar figuras interactivamente:

```powershell
python .\Survival-Churn-app\models.py --show-plots
```

- Guardar en una carpeta personalizada:

```powershell
python .\Survival-Churn-app\models.py --save-plots --outdir results/figuras
```

Notas:
- Por defecto las figuras se guardan en `plots/` si usas `--save-plots`.
- Si quieres ejecutar el código desde otra carpeta: el script busca el CSV con ruta relativa al propio archivo (`Telco_customer_churn.csv`).
- Para un uso interactivo más cómodo, ejecuta el archivo en una terminal dentro de la carpeta del proyecto o usa `--show-plots`.

--------------------------

Despliegue en Render (Streamlit):
- Build Command: `pip install -r requirements.txt`
- Start Command:
	`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

Notas:
- Render exige escuchar en `0.0.0.0` y en el puerto indicado por `PORT` (por defecto 10000).
- También puedes usar el archivo `render.yaml` incluido en la raíz del proyecto.
