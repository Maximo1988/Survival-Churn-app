import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

st.set_page_config(layout="wide", page_title="Survival Churn App", initial_sidebar_state="expanded")

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
    results = {}
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            cat1, cat2 = cats[i], cats[j]
            mask1 = df[group_col] == cat1
            mask2 = df[group_col] == cat2
            res = logrank_test(df[mask1][duration_col], df[mask2][duration_col],
                               event_observed_A=df[mask1][event_col], event_observed_B=df[mask2][event_col])
            p = res.p_value
            # Try to compute HR via Cox on the two groups (cat2 vs cat1)
            hr = hr_lower = hr_upper = None
            try:
                subset = df[df[group_col].isin([cat1, cat2])][[duration_col, event_col, group_col]].dropna()
                subset = subset.copy()
                subset['__grp__'] = (subset[group_col] == cat2).astype(int)
                cph_tmp = CoxPHFitter()
                cph_tmp.fit(subset[[duration_col, event_col, '__grp__']], duration_col=duration_col, event_col=event_col)
                hr = float(cph_tmp.hazard_ratios_.values[0])
                summ = cph_tmp.summary
                hr_lower = float(summ['exp(coef) lower 95%'].iloc[0])
                hr_upper = float(summ['exp(coef) upper 95%'].iloc[0])
            except Exception:
                pass
            results[f"{cat1} vs {cat2}"] = {'p_value': p, 'HR': hr, 'HR_lower': hr_lower, 'HR_upper': hr_upper}
    return results

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
        return cph, hr, concord, None
    except Exception as e:
        # Return the exception string to explain why the model failed
        return None, None, None, str(e)

# --- App UI ---
# CSS para reducir ancho del sidebar
st.markdown("""
<style>
    /* Reducir ancho del sidebar al 80% (de 37rem a 30rem) */
    [data-testid="stSidebar"] {
        min-width: 30rem;
        max-width: 30rem;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 30rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Análisis de Churn — Supervivencia y Riesgo")
df = load_data()

# Sidebar: selección de variables significativas (preseleccionadas)
with st.sidebar:
    st.header("Variables significativas")
    recommended = ['Contract', 'Online Security', 'Tech Support', 'Payment Method',
                   'Online Backup', 'Device Protection', 'Partner', 'Dependents',
                   'Internet Service', 'Streaming Movies', 'Streaming TV',
                   'Paperless Billing', 'Senior Citizen', 'Multiple Lines']

    # Column groups that are NOT predictors
    id_cols = ['CustomerID']
    geo_cols = ['Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude']
    # 'Churn Value' se usa como variable de evento, no como predictor
    non_predictor_cols = ['Count', 'Churn Reason', 'Churn Label', 'Churn Score', 'CLTV', 'Churn Value']
    disabled_cols = set(id_cols + geo_cols + non_predictor_cols)

    # Build the list of selectable variables that ARE used in the models
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Only show the recommended model variables (if present and not disabled)
    available = [c for c in recommended if c in obj_cols and c not in disabled_cols]

    # Ensure uniqueness and preserve order
    available = list(dict.fromkeys(available))
    default_sel = [x for x in recommended[:3] if x in available]

    selected = st.multiselect("Seleccion de variables:", options=available, default=default_sel)

    st.markdown("---")
    st.write("Instrucciones:")
    st.write("Para cada variable seleccionada, verás las curvas de supervivencia siguientes, según corresponda:")
    st.markdown("• Kaplan-Meier\n• Log-Rank\n• Regresión COX")
    include_numeric = st.checkbox("Incluir 'Monthly Charges' en el modelo COX", value=True)

    # Collapsible panel to show non-predictor/identifier/geographic columns only on demand
    with st.expander("Variables deshabilitadas", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write("**Identificadores**")
            for col in id_cols:
                if col in df.columns:
                    st.checkbox(f"{col} (identificadores)", value=False, disabled=True)
        with col_b:
            st.write("**Geográficas**")
            for col in geo_cols:
                if col in df.columns:
                    st.checkbox(f"{col} (geograficas)", value=False, disabled=True)
        with col_c:
            st.write("**No predictoras**")
            # Map 'Count' to a clearer label if present
            for col in non_predictor_cols:
                if col in df.columns:
                    label = col
                    if col == 'Count':
                        label = 'Count (Customer Count)'
                    st.checkbox(f"{label} (No predictoras)", value=False, disabled=True)
        st.caption("Nota: estas columnas no se consideran predictoras para los modelos")

# Main area: mostrar tabla y análisis por variable
st.subheader("Vista previa e inicial de los datos")
st.dataframe(df.head(5))

if not selected:
    st.warning("Selecciona al menos una variable en la barra lateral para ver los análisis.")
else:
    # Selector central único para todos los modelos
    choice_cols = st.columns([1,2,1])
    with choice_cols[1]:
        view_choice = st.radio("Seleccionar modelo:", options=["KM","Log‑Rank","Cox"], index=0, horizontal=True, key="view_choice_global")
    
    st.markdown("---")
    
    for var in selected:
        st.markdown(f"### Variable: **{var}**")

        def render_km():
            col_title, col_info = st.columns([5,1])
            with col_title:
                st.write("**Kaplan–Meier (KM)**")
            with col_info:
                with st.expander("Info KM", expanded=False):
                    st.write("KM analizará el tiempo que transcurre hasta que un cliente abandonará el servicio")
            km_results = compute_km(df, 'Tenure Months', 'Churn Value', var)
            try:
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
                        st.markdown("<h3 style='color:red'>No se pudo ajustar Kaplan–Meier: datos insuficientes</h3>", unsafe_allow_html=True)
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
                        st.markdown("<h3 style='color:red'>No se pudo ajustar Kaplan–Meier: datos insuficientes</h3>", unsafe_allow_html=True)
                    else:
                        ax.set_xlabel('Tiempo (meses)')
                        ax.set_ylabel('Probabilidad de supervivencia')
                        ax.set_title(f'Curva KM por {var}')
                        ax.legend()
                        st.pyplot(fig)

                # Texto resumen KM
                if plotted:
                    try:
                        summaries = []
                        last_surv = []
                        for grp, res in km_results.items():
                            if res is None or res.empty:
                                continue
                            surv = res['survival']
                            timeline = res['timeline']
                            # Mediana aproximada: primer tiempo donde supervivencia <= 0.5
                            median_time = None
                            if (surv <= 0.5).any():
                                median_time = timeline[surv <= 0.5].iloc[0]
                            last_surv.append((grp, float(surv.iloc[-1]), float(timeline.iloc[-1])))
                            if median_time is None:
                                summaries.append(f"Grupo {grp}: no alcanza 50% de abandono en el periodo observado.")
                            else:
                                summaries.append(f"Grupo {grp}: mediana de abandono ≈ {median_time:.0f} meses.")

                        st.markdown("**Resultado KM:**")
                        if last_surv:
                            best = sorted(last_surv, key=lambda x: x[1], reverse=True)[0]
                            st.write(f"Mayor retención al final del periodo: {best[0]} (supervivencia ≈ {best[1]:.2f} a {best[2]:.0f} meses).")
                        for line in summaries:
                            st.write(line)
                    except Exception:
                        st.info("No fue posible generar el resumen textual de KM para esta variable.")
            except Exception as e:
                st.markdown(f"<h3 style='color:red'>No se pudo ajustar Kaplan–Meier</h3>\n\n**Razón:** {e}", unsafe_allow_html=True)


        def render_cox():
            col_title, col_info = st.columns([5,1])
            with col_title:
                st.write("**Cox Proportional Hazards (COX)**")
            with col_info:
                with st.expander("Info COX", expanded=False):
                    st.write("COX te mostrará la probabilidad de que un cliente permanezca en la empresa a lo largo del tiempo, pero tomando en cuenta el riesgo instantáneo de diversas variables que influyan en su abandono")
            cph, hr, concord, err = fit_cox_for_var(df, 'Tenure Months', 'Churn Value', var, include_numeric=include_numeric)
            if cph is None:
                reason = err or "No fue posible ajustar el modelo Cox con las columnas seleccionadas (p. ej. muy pocas observaciones o colinealidad)."
                st.markdown(f"<h3 style='color:red'>Modelo COX no pudo ser obtenido</h3>\n\n**Razón:** {reason}", unsafe_allow_html=True)
                return
            try:
                summary = cph.summary
                hr_df = summary[["exp(coef)"]].rename(columns={"exp(coef)": "HR"})

                # percent change and formatted strings (only HR general)
                hr_pct_num = (hr_df['HR'] - 1) * 100
                display_df = pd.DataFrame({
                    'HR (%)': hr_pct_num.map(lambda x: f"{x:.2f}%"),
                }, index=hr_df.index)

                # Show table (only HR) and concordance
                st.dataframe(display_df)
                if concord is not None:
                    st.metric("Concordance (C-index)", f"{concord*100:.2f}%")
                
                # Forest Plot (HR + IC 95%)
                try:
                    conf_ints = cph.confidence_intervals_
                    if conf_ints is not None and not conf_ints.empty:
                        ci_lower_col = conf_ints.columns[0]
                        ci_upper_col = conf_ints.columns[1]
                        plot_df = hr_df.join(conf_ints[[ci_lower_col, ci_upper_col]], how='left')
                        plot_df = plot_df.dropna().sort_values('HR', ascending=False).head(8)

                        if not plot_df.empty:
                            if PLOTLY_AVAILABLE:
                                fig = go.Figure()
                                for name, row in plot_df.iterrows():
                                    hr_val = float(row['HR'])
                                    lower = float(row[ci_lower_col])
                                    upper = float(row[ci_upper_col])
                                    color = '#ff6b6b' if hr_val > 1 else '#51cf66'

                                    fig.add_trace(go.Scatter(
                                        x=[lower, upper],
                                        y=[name, name],
                                        mode='lines',
                                        line=dict(color=color, width=2),
                                        showlegend=False,
                                        hovertemplate=f"{name}<br>IC: [{lower:.3f}, {upper:.3f}]<extra></extra>"
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=[hr_val],
                                        y=[name],
                                        mode='markers',
                                        marker=dict(size=10, color=color),
                                        showlegend=False,
                                        hovertemplate=f"{name}<br>HR: {hr_val:.3f}<extra></extra>"
                                    ))

                                fig.add_vline(x=1, line_dash="dash", line_color="gray")
                                fig.update_layout(
                                    title_text="",
                                    xaxis_title="",
                                    yaxis_title="",
                                    hovermode='closest',
                                    height=350,
                                    template='plotly_white',
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                y_pos = range(len(plot_df))
                                for i, (name, row) in enumerate(plot_df.iterrows()):
                                    hr_val = float(row['HR'])
                                    lower = float(row[ci_lower_col])
                                    upper = float(row[ci_upper_col])
                                    color = '#ff6b6b' if hr_val > 1 else '#51cf66'
                                    ax.errorbar(hr_val, i, xerr=[[hr_val - lower], [upper - hr_val]],
                                               fmt='o', color=color, markersize=8, linewidth=2)
                                ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels([str(n) for n in plot_df.index])
                                ax.set_xlabel("")
                                ax.set_title("")
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                        else:
                            st.info("No hay suficientes datos para el Forest Plot.")
                    else:
                        st.info("No hay intervalos de confianza disponibles para el Forest Plot.")
                except Exception as e:
                    st.warning(f"No se pudo generar Forest Plot: {e}")
                
                # Top covariables con interpretación simplificada
                try:
                    top = hr_df.sort_values('HR', ascending=False).head(5)
                    st.markdown("#### Top covariables por efecto en abandono (HR)")
                    
                    for idx, (name, row) in enumerate(top.iterrows(), 1):
                        hr_val = float(row['HR'])
                        hr_pct = (hr_val - 1) * 100
                        
                        # Determine effect direction
                        if hr_val > 1:
                            color = "#ff6b6b"
                            message = f"El valor de riesgo de abandono se aumenta a un: {hr_pct:.2f}%"
                        else:
                            color = "#51cf66"
                            message = f"El valor de riesgo de abandono se reduce a un: {abs(hr_pct):.2f}%"
                        
                        # Display with simple interpretation
                        st.markdown(f"**{idx}. {name}**")
                        st.markdown(f"<div style='background-color:{color}22; padding:10px; border-left:4px solid {color}; border-radius:5px;'>"
                                  f"{message}"
                                  f"</div>", unsafe_allow_html=True)
                    
                    # Conclusión general
                    st.markdown("---")
                    st.markdown("**Conclusión COX para esta variable:**")
                    
                    accelerators = top[top['HR'] > 1]
                    retarders = top[top['HR'] < 1]
                    
                    conclusion = f"En el análisis de '{var}':\n"
                    
                    if len(accelerators) > 0:
                        acc_vars = ", ".join([f"**{n}**" for n in accelerators.index[:3]])
                        conclusion += f"\n🔴 **Variables que ACELERAN el abandono:** {acc_vars}\n"
                    
                    if len(retarders) > 0:
                        ret_vars = ", ".join([f"**{n}**" for n in retarders.index[:3]])
                        conclusion += f"\n🟢 **Variables que RETRASAN el abandono:** {ret_vars}\n"
                    
                    if len(accelerators) == 0 and len(retarders) == 0:
                        conclusion += "\nNo se encontraron variables con efecto significativo.\n"
                    
                    conclusion += f"\nEl modelo COX alcanza una precisión de predicción del {concord*100:.2f}% (C-index), indicando que es {'altamente' if concord > 0.7 else 'moderadamente' if concord > 0.6 else 'débilmente'} efectivo para predecir abandono en base a estos factores."
                    
                    st.info(conclusion)
                    
                except Exception as e:
                    st.warning(f"No se pudo generar análisis detallado de top covariables: {e}")
            except Exception as e:
                st.markdown(f"<h3 style='color:red'>No se pudo mostrar resultados COX</h3>\n\n**Razón:** {e}", unsafe_allow_html=True)

        def render_logrank():
            # Log-Rank: comparaciones por pares
            try:
                col_title, col_info = st.columns([5,1])
                with col_title:
                    st.write("**Log‑Rank (comparaciones por pares)**")
                with col_info:
                    with st.expander("Info Log‑Rank", expanded=False):
                        st.write("LOG-RANK evaluará la probabilidad de que un cliente permanezca activo a lo largo del tiempo y te dirá si existen diferencias importantes en el tiempo de retención entre dos o más grupos de clientes")
                
                # KM curve (graphical view)
                km_results = compute_km(df, 'Tenure Months', 'Churn Value', var)
                if not km_results:
                    st.markdown("<h3 style='color:red'>No se pudo ajustar Kaplan–Meier: datos insuficientes</h3>", unsafe_allow_html=True)
                else:
                    # Plot
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure()
                        for grp, res in km_results.items():
                            if res is None:
                                continue
                            fig.add_trace(go.Scatter(x=res['timeline'], y=res['survival'], mode='lines', 
                                                   name=f"Grupo: {grp}",
                                                   hovertemplate=f'Grupo: {grp}<br>Mes: %{{x}}<br>Supervivencia: %{{y:.3f}}<extra></extra>'))
                        fig.update_layout(title=f'Curva KM - {var}', xaxis_title='Tiempo (meses)', yaxis_title='Probabilidad de supervivencia', template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(10,6))
                        for grp, res in km_results.items():
                            if res is None:
                                continue
                            ax.step(res['timeline'], res['survival'], where='post', label=f"Grupo: {grp}")
                        ax.set_xlabel('Tiempo (meses)')
                        ax.set_ylabel('Probabilidad de supervivencia')
                        ax.set_title(f'Curva KM - {var}')
                        ax.legend()
                        st.pyplot(fig)

                # Table with pairwise comparisons (Log-Rank)
                pvals = compute_logrank_pairs(df, 'Tenure Months', 'Churn Value', var)
                if not pvals:
                    st.write("No se pudo calcular Log-Rank")
                else:
                    rows = []
                    p_list = []
                    for comp, vals in pvals.items():
                        p = vals.get('p_value')
                        hr = vals.get('HR')
                        if hr is not None:
                            hr_pct = (hr - 1) * 100
                            hr_str = f"{hr_pct:.2f}%"
                        else:
                            hr_str = "N/A"
                        rows.append((comp, hr_str))
                        if p is not None:
                            p_list.append((comp, p, hr))

                    p_df = pd.DataFrame(rows, columns=['Comparación','Porcentaje (%)'])
                    st.dataframe(p_df)

                    # Conclusion
                    sig_pairs = [t for t in p_list if t[1] is not None and t[1] < 0.05]
                    if sig_pairs:
                        best = sorted(sig_pairs, key=lambda x: x[1])[0]
                        comp_best, p_best, hr_best = best
                        hr_text = "N/A"
                        if hr_best is not None:
                            hr_text = f"{(hr_best-1)*100:.2f}%"
                        st.success(f"✓ Se detectan diferencias significativas en {var}.\n\nComparación más relevante: {comp_best} con efecto de {hr_text}.")
                    else:
                        st.info(f"ℹ No se detectan diferencias significativas entre los grupos de {var}.")
            except Exception as e:
                st.markdown(f"<h3 style='color:red'>No se pudo calcular Log‑Rank</h3>\n\n**Razón:** {e}", unsafe_allow_html=True)

        if view_choice == "KM":
            render_km()
        elif view_choice == "Log‑Rank":
            render_logrank()
        elif view_choice == "Cox":
            render_cox()

        st.markdown('---')

st.markdown('---')
# Sugerencias contextuales según el modelo seleccionado
if view_choice == "KM":
    st.write('Sugerencia: Log-Rank te muestra comparaciones por pares para cada variable seleccionada.')
elif view_choice == "Log‑Rank":
    st.write('Sugerencia: COX te mostrará qué variables impulsarán el riesgo instantáneo de abandono.')
