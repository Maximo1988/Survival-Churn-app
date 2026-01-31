import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Survival Churn App")

SCRIPT_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(SCRIPT_DIR, "Telco_customer_churn.csv")

@st.cache_data
def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    return df

@st.cache_data
def compute_km(df, duration_col, event_col, group_col):
    kmf = KaplanMeierFitter()
    results = {}
    for grp in sorted(df[group_col].dropna().unique()):
        mask = df[group_col] == grp
        try:
            kmf.fit(df[mask][duration_col], df[mask][event_col], label=str(grp))
            results[grp] = kmf.survival_function_.reset_index().rename(columns={"timeline": "timeline", kmf.survival_function_.columns[0]: "survival"})
        except Exception:
            results[grp] = None
    return results

@st.cache_data
def compute_logrank_pairs(df, duration_col, event_col, group_col):
    cats = sorted(df[group_col].dropna().unique())
    pvals = {}
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            cat1, cat2 = cats[i], cats[j]
            mask1 = df[group_col] == cat1
            mask2 = df[group_col] == cat2
            res = logrank_test(df[mask1][duration_col], df[mask2][duration_col], event_observed_A=df[mask1][event_col], event_observed_B=df[mask2][event_col])
            pvals[f"{cat1} vs {cat2}"] = res.p_value
    return pvals

@st.cache_data
def fit_cox_for_var(df, duration_col, event_col, var, include_numeric=False):
    d = df[[duration_col, event_col, var]].copy()
    # One-hot encode categorical var
    dummies = pd.get_dummies(d[var], prefix=var.replace(' ', '_'), drop_first=True)
    X = pd.concat([d[[duration_col, event_col]].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    # Optionally include Monthly Charges if present and requested
    if include_numeric and 'Monthly Charges' in df.columns:
        X = pd.concat([X, df[['Monthly Charges']].reset_index(drop=True)], axis=1)
    # Drop NA and ensure numeric
    X = X.dropna()
    try:
        cph = CoxPHFitter()
        cph.fit(X, duration_col=duration_col, event_col=event_col)
        hr = cph.hazard_ratios_.to_frame(name='HR')
        concord = cph.concordance_index_
        return cph, hr, concord
    except Exception as e:
        return None, None, None

# --- App UI ---
st.title("Análisis de Churn — Supervivencia y Riesgo")
df = load_data()

# Sidebar: selección de variables significativas (preseleccionadas)
with st.sidebar:
    st.header("Variables significativas")
    recommended = ['Contract', 'Online Security', 'Tech Support', 'Payment Method',
                   'Online Backup', 'Device Protection', 'Partner', 'Dependents',
                   'Internet Service', 'Streaming Movies', 'Streaming TV',
                   'Paperless Billing', 'Senior Citizen', 'Multiple Lines']
    available = [c for c in df.select_dtypes(include=['object']).columns.tolist() if c in recommended or c in df.columns]
    # Ensure uniqueness and preserve order
    available = list(dict.fromkeys(available))
    selected = st.multiselect("Selecciona variables (una o varias):", options=available, default=recommended[:3])
    st.markdown("---")
    st.write("Instrucciones:")
    st.write("Selecciona una o varias variables para analizar. Para cada variable, se mostrará: (1) curva de supervivencia (Kaplan–Meier), (2) prueba Log-Rank entre categorías y (3) modelo Cox para HR y concordancia.")
    include_numeric = st.checkbox("Incluir 'Monthly Charges' en el modelo Cox", value=False)

# Main area: mostrar tabla y análisis por variable
st.subheader("Vista de datos")
st.dataframe(df.head(5))

if not selected:
    st.warning("Selecciona al menos una variable en la barra lateral para ver los análisis.")
else:
    for var in selected:
        st.markdown(f"### Variable: **{var}**")
        cols = st.columns([2, 1])
        # Left: plots
        with cols[0]:
            st.write("**Curva de supervivencia (KM) por categoría**")
            km_results = compute_km(df, 'Tenure Months', 'Churn Value', var)

            # KM plot: Plotly si está disponible, sino Matplotlib
            plotted = False
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                for grp, res in km_results.items():
                    if res is None:
                        continue
                    fig.add_trace(go.Scatter(x=res['timeline'], y=res['survival'], mode='lines', name=str(grp),
                                             hovertemplate=f'Grupo: {grp}<br>Mes: %{{x}}<br>Supervivencia: %{{y:.3f}}<extra></extra>'))
                    plotted = True
                if not plotted:
                    st.write("No hay suficientes datos para ajustar Kaplan–Meier.")
                else:
                    fig.update_layout(title=f'Curva KM por {var}', xaxis_title='Tiempo (meses)', yaxis_title='Probabilidad de supervivencia', template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(8,4))
                for grp, res in km_results.items():
                    if res is None:
                        continue
                    ax.step(res['timeline'], res['survival'], where='post', label=str(grp))
                    plotted = True
                if not plotted:
                    st.write("No hay suficientes datos para ajustar Kaplan–Meier.")
                else:
                    ax.set_xlabel('Tiempo (meses)')
                    ax.set_ylabel('Probabilidad de supervivencia')
                    ax.set_title(f'Curva KM por {var}')
                    ax.legend()
                    st.pyplot(fig)

            # Churn por mes (por categoría) — porcentajes respecto al total de churn de la categoría (interactive Plotly stacked bar)
            st.write("**Distribución de churn por Tenure (por mes)**")
            churn = df[df['Churn Value']==1]
            cats = sorted(churn[var].dropna().unique())
            max_t = min(72, int(df['Tenure Months'].max()))
            months = list(range(0, max_t+1))

            frames = []
            for cat in cats:
                cnts = churn[churn[var]==cat]['Tenure Months'].value_counts().reindex(months, fill_value=0).sort_index()
                if cnts.sum() > 0:
                    pct = (cnts / cnts.sum() * 100).round(2)
                else:
                    pct = cnts
                df_pct = pd.DataFrame({'month': months, 'pct': pct.values, 'category': cat})
                frames.append(df_pct)

            if frames:
                df_long = pd.concat(frames, ignore_index=True)
                if PLOTLY_AVAILABLE:
                    fig2 = px.bar(df_long, x='month', y='pct', color='category', labels={'month':'Tenure (meses)','pct':'% churn por mes','category':var},
                                  title=f'Distribución de churn por Tenure por {var}', template='plotly_white')
                    fig2.update_layout(barmode='stack', legend=dict(orientation='h', y=-0.2))
                    fig2.update_traces(hovertemplate='%{y:.2f}% %{fullData.name}<br>Mes: %{x}<extra></extra>')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    fig2, ax2 = plt.subplots(figsize=(8,3))
                    bottom = None
                    for cat in cats:
                        cnts = churn[churn[var]==cat]['Tenure Months'].value_counts().reindex(months, fill_value=0).sort_index()
                        pct = cnts / cnts.sum() * 100 if cnts.sum()>0 else cnts
                        if bottom is None:
                            ax2.bar(pct.index, pct.values, label=str(cat))
                            bottom = pct.values
                        else:
                            ax2.bar(pct.index, pct.values, bottom=bottom, label=str(cat))
                            bottom = bottom + pct.values
                    ax2.set_xlabel('Tenure (meses)')
                    ax2.set_ylabel('% churn por mes (por categoría)')
                    ax2.set_xlim(0, max_t)
                    ax2.legend(fontsize=8)
                    st.pyplot(fig2)
            else:
                st.write("No hay datos de churn para esta variable.")

            # Mostrar cómo se calculó: método
            st.info("Método: Kaplan–Meier para estimar supervivencia; prueba Log-Rank para comparar curvas; modelo Cox PH para estimar HR y concordancia.")

        # Right: stats (log-rank p-values and Cox HR + concordance)
        with cols[1]:
            st.write("**Resultados estadísticos**")
            # Log-rank pairwise p-values
            pvals = compute_logrank_pairs(df, 'Tenure Months', 'Churn Value', var)
            if pvals:
                p_df = pd.DataFrame(list(pvals.items()), columns=['Comparación','p-value']).sort_values('p-value')
                st.write("Log-Rank (p-values) — comparaciones por pares")
                st.dataframe(p_df)
                sig = p_df[p_df['p-value'] < 0.05]
                if not sig.empty:
                    st.success(f"Se detectaron diferencias significativas en {len(sig)} comparaciones (p<0.05).")
                else:
                    st.info("No se detectaron diferencias significativas (p>=0.05) entre categorías para esta variable.")
            else:
                st.write("No se pudieron calcular p-values (muy pocas categorías o datos).")

            # Cox model for HR
            cph, hr, concord = fit_cox_for_var(df, 'Tenure Months', 'Churn Value', var, include_numeric=include_numeric)
            if cph is None:
                st.write("No fue posible ajustar un modelo Cox con las columnas seleccionadas (p. ej. muy pocas observaciones o colinealidad).")
            else:
                st.write("Cox PH — Hazard Ratios (HR) con IC 95%")
                # Build HR table with confidence intervals
                try:
                    summary = cph.summary
                    hr_df = summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]].rename(columns={"exp(coef)": "HR", "exp(coef) lower 95%": "HR_lower", "exp(coef) upper 95%": "HR_upper"})
                    hr_df = hr_df.round(3)
                    st.dataframe(hr_df)
                except Exception:
                    # fallback to simple HR if summary not available
                    if hr is not None:
                        st.dataframe(hr.round(3))

                # Show concordance as a metric
                if concord is not None:
                    st.metric("Concordance (C-index)", f"{concord:.3f}")
                st.markdown("**Nota:** HR > 1 indica mayor riesgo de churn comparado con la categoría de referencia (drop_first).")

                # Present top covariables por HR
                try:
                    top = hr_df.sort_values('HR', ascending=False).head(5)
                    st.write("Top covariables por HR")
                    st.table(top)
                except Exception:
                    pass

st.markdown('---')
st.write('Sugerencia: selecciona una sola variable para análisis más claro, o varias para comparar múltiples variables una a una.')
