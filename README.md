# Survival-Churn-Analysis

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
- Si quieres ejecutar el código desde otra carpeta, no te preocupes: el script busca el CSV con ruta relativa al propio archivo (`Telco_customer_churn.csv`).
- Para un uso interactivo más cómodo, ejecuta el archivo en una terminal dentro de la carpeta del proyecto o usa `--show-plots`.
