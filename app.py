# app.py
import json
import io
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px
from optimizer import run_optimizer

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(page_title="Optimizador de Producción y Precios", layout="wide")
st.title("Optimizador de Producción y Precios — Comparativa de empresas (Simplex)")


# ============================================
# UTILIDAD PARA PARSEAR JSON DE EMPRESAS
# ============================================
def parse_companies_from_json(raw_json):
    """
    Espera JSON con estructura:
    {
      "companies": [
         {
            "name": "...",
            "materials": [...],
            "products": [...]
         }
      ]
    }
    """
    j = json.loads(raw_json)
    companies = []

    for comp in j.get("companies", []):
        # Materias primas
        mats = comp.get("materials", [])
        mat_df = pd.DataFrame(mats)

        # Productos
        prods = comp.get("products", [])
        prod_rows = []
        for p in prods:
            row = {
                "name": p.get("name"),
                "unit_cost": float(p.get("unit_cost", 0.0)),
                "demand": p.get("demand_a", None),
                "demand_b": p.get("demand_b", None),
                "price_min": p.get("price_min", None),
                "price_max": p.get("price_max", None)
            }
            # consumos por materia
            materials_consumption = p.get("materials", {}) or {}
            for mname, mval in materials_consumption.items():
                row[mname] = float(mval)
            prod_rows.append(row)

        prod_df = pd.DataFrame(prod_rows)

        companies.append({
            "name": comp.get("name", "Empresa"),
            "materials_df": mat_df,
            "products_df": prod_df,
            "raw": comp
        })

    return companies


# ============================================
# UTILIDAD PARA CONSTRUIR PROBLEMA LP
# ============================================
def build_lp_problem(materials_df, products_df, price_mode="promedio", override_profit=None):
    """
    Construye los coeficientes para linprog.
    price_mode ∈ {promedio, min, max}
    """
    # Materias primas
    mat_names = materials_df["name"].tolist()
    if "available" in materials_df.columns:
        b = materials_df["available"].to_numpy(dtype=float)
    elif "Disponibilidad" in materials_df.columns:
        b = materials_df["Disponibilidad"].to_numpy(dtype=float)
    else:
        b = materials_df.iloc[:, 1].to_numpy(dtype=float)

    # Productos
    prod_names = products_df["name"].tolist()
    n = len(prod_names)

    # Matriz A de consumos
    A = []
    for m in mat_names:
        if m in products_df.columns:
            A.append(products_df[m].fillna(0).to_numpy(dtype=float))
        else:
            A.append(np.zeros(n))
    A = np.array(A)

    # Ganancias
    profits = np.zeros(n)
    for i in range(n):
        row = products_df.iloc[i]
        unit_cost = float(row.get("unit_cost", 0.0))

        if override_profit is not None:
            profits[i] = float(override_profit[i])
        else:
            pmin = row.get("price_min")
            pmax = row.get("price_max")

            if pmin is None and pmax is None:
                profits[i] = -unit_cost
            else:
                if price_mode == "promedio":
                    price = ((pmin if pmin else pmax) + (pmax if pmax else pmin)) / 2
                elif price_mode == "min":
                    price = pmin if pmin else pmax
                else:
                    price = pmax if pmax else pmin
                profits[i] = price - unit_cost

    # Bounds por producto
    bounds = []
    for i in range(n):
        d = products_df.iloc[i].get("demand")
        if d is None:
            bounds.append((0, None))
        else:
            bounds.append((0, float(d)))

    # linprog minimiza => usamos -profits para maximizar
    c = -profits

    return c, A, b, bounds, prod_names, profits, mat_names


def solve_lp(c, A, b, bounds):
    return linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method="highs")


# ============================================
# BARRA LATERAL — ENTRADAS
# ============================================
st.sidebar.header("Entradas y ajustes")
example_data = st.sidebar.checkbox("Cargar JSON de ejemplo (comparativa)")
uploaded_file = st.sidebar.file_uploader("Subir archivo JSON con estructura 'companies'", type=["json"])
raw_json_text = st.sidebar.text_area("O pegar JSON aquí", height=180)

st.sidebar.subheader("Ajustes de optimización")
price_mode = st.sidebar.selectbox("Modo de precio", ["promedio", "min", "max"])
use_override_profit = st.sidebar.checkbox("Sobrescribir ganancias manualmente")

# JSON de ejemplo
EXAMPLE_JSON = """
{
  "companies": [
    {
      "name": "Bastian Solutions",
      "materials": [
        {"name": "Caja", "available": 5000},
        {"name": "Contenedor", "available": 2000}
      ],
      "products": [
        {
          "name": "SKU1",
          "unit_cost": 2,
          "materials": {"Caja": 1, "Contenedor": 0.2},
          "demand_a": 10000,
          "price_min": 3,
          "price_max": 10
        },
        {
          "name": "SKU2",
          "unit_cost": 5,
          "materials": {"Caja": 2, "Contenedor": 0.5},
          "demand_a": 5000,
          "price_min": 6,
          "price_max": 20
        }
      ]
    },
    {
      "name": "Textindustria",
      "materials": [
        {"name": "Tela", "available": 20000},
        {"name": "Hilo", "available": 10000}
      ],
      "products": [
        {
          "name": "Blusa A",
          "unit_cost": 8,
          "materials": {"Tela": 2.5, "Hilo": 0.1},
          "demand_a": 3000,
          "price_min": 10,
          "price_max": 50
        },
        {
          "name": "Blusa B",
          "unit_cost": 10,
          "materials": {"Tela": 3, "Hilo": 0.15},
          "demand_a": 2000,
          "price_min": 12,
          "price_max": 60
        }
      ]
    }
  ]
}
"""

# Selección de fuente JSON
raw_json = None
if uploaded_file:
    raw_json = uploaded_file.read().decode("utf-8")
elif raw_json_text.strip() != "":
    raw_json = raw_json_text
elif example_data:
    raw_json = EXAMPLE_JSON

if raw_json is None:
    st.info("Cargue o pegue JSON para iniciar.")
    st.stop()


# ============================================
# PARSEAR EMPRESAS
# ============================================
try:
    companies = parse_companies_from_json(raw_json)
except Exception as e:
    st.error(f"Error al parsear JSON: {e}")
    st.stop()

# Selector empresa
company_names = [c["name"] for c in companies]
selected = st.sidebar.selectbox("Seleccione empresa", company_names)
company = companies[company_names.index(selected)]

st.subheader("Información de la empresa seleccionada")
st.write(company["name"])
st.write("**Materias primas:**")
st.dataframe(company["materials_df"])
st.write("**Productos:**")
st.dataframe(company["products_df"])


# ============================================
# SOBRESCRIBIR GANANCIAS
# ============================================
override_profit = None
if use_override_profit:
    st.subheader("Ganancias manuales por producto")
    prod_names = company["products_df"]["name"].tolist()
    override_profit = []
    cols = st.columns(len(prod_names))
    for i, pname in enumerate(prod_names):
        with cols[i]:
            val = st.number_input(f"G. {pname}", value=10.0, key=f"g_{i}")
            override_profit.append(val)


# ============================================
# OPTIMIZACIÓN INDIVIDUAL
# ============================================
st.markdown("---")
if st.button("Ejecutar optimización (empresa seleccionada)"):

    try:
        c, A, b, bounds, prod_names, profits, mat_names = build_lp_problem(
            company["materials_df"], company["products_df"],
            price_mode=price_mode, override_profit=override_profit
        )
        res = solve_lp(c, A, b, bounds)

        if not res.success:
            st.error("No se encontró solución óptima.")
            st.write(res.message)
            st.stop()

        production = np.round(res.x, 6)
        total_profit = float(np.dot(profits, production))

        st.success("Optimización resuelta.")
        results_df = pd.DataFrame({
            "Producto": prod_names,
            "Cantidad óptima": production,
            "Ganancia unitaria": profits,
            "Contribución total": profits * production
        })
        st.dataframe(results_df)

        st.metric("Ganancia total", f"${total_profit:,.2f}")

        # Graficas
        fig1 = px.bar(results_df, x="Producto", y="Cantidad óptima")
        st.plotly_chart(fig1, use_container_width=True)

    except Exception as e:
        st.error(f"Error en la optimización: {e}")


# ============================================
# COMPARATIVA ENTRE TODAS LAS EMPRESAS
# ============================================
st.markdown("---")
st.subheader("Comparativa entre todas las empresas")

if st.button("Ejecutar comparativa general"):
    summary = []
    details = []

    for comp in companies:
        try:
            c, A, b, bounds, prod_names, profits, mat_names = build_lp_problem(
                comp["materials_df"], comp["products_df"], price_mode=price_mode
            )
            res = solve_lp(c, A, b, bounds)

            if res.success:
                production = np.round(res.x, 6)
                total_profit = float(np.dot(profits, production))

                summary.append({
                    "Empresa": comp["name"],
                    "Ganancia": total_profit,
                    "Producción total": float(production.sum())
                })

                details.append({
                    "Empresa": comp["name"],
                    "Productos": prod_names,
                    "Producción óptima": production.tolist()
                })
            else:
                summary.append({
                    "Empresa": comp["name"],
                    "Ganancia": None,
                    "Producción total": None
                })

        except:
            summary.append({
                "Empresa": comp["name"],
                "Ganancia": None,
                "Producción total": None
            })

    df_summary = pd.DataFrame(summary)
    st.dataframe(df_summary)

    fig = px.bar(df_summary, x="Empresa", y="Ganancia")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detalles por empresa")
    for d in details:
        st.write(f"**{d['Empresa']}**")
        st.dataframe(pd.DataFrame({
            "Producto": d["Productos"],
            "Producción óptima": d["Producción óptima"]
        }))


# ============================================
# COMPARATIVA OPTIMIZER vs FLEXSIM
# ============================================
st.markdown("---")
st.subheader("Comparativa Optimizer vs FlexSim")

try:
    with open("sample_data.json", "r") as f:
        flex_data = json.load(f)
except Exception as e:
    st.error(f"No se pudo cargar sample_data.json: {e}")
    st.stop()

if "flexsim" not in flex_data:
    st.error("sample_data.json debe contener la clave 'flexsim'")
    st.stop()

df_flex = pd.DataFrame(flex_data["flexsim"])
st.write("**Datos reales (FlexSim):**")
st.dataframe(df_flex)

# Resolver optimizador para misma empresa
try:
    c, A, b, bounds, prod_names, profits, mat_names = build_lp_problem(
        company["materials_df"], company["products_df"], price_mode=price_mode
    )
    res = solve_lp(c, A, b, bounds)

    if not res.success:
        st.error("No se pudo resolver optimización para comparativa.")
        st.stop()

    production_opt = np.round(res.x, 6)

    df_opt = pd.DataFrame({
        "Producto": prod_names,
        "Producción óptima": production_opt
    })

    st.write("**Resultados del Optimizer (Simplex):**")
    st.dataframe(df_opt)

    # Comparativa
    df_cmp = df_opt.merge(df_flex, on="Producto", how="inner")
    df_cmp["Diferencia"] = df_cmp["Producción óptima"] - df_cmp["real_production"]

    st.subheader("Comparativa final")
    st.dataframe(df_cmp)

    fig = px.bar(df_cmp, x="Producto",
                 y=["Producción óptima", "real_production"],
                 barmode="group")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error en comparativa Optimizer/FlexSim: {e}")

# ============================================
# EXPLICACIÓN FINAL PARA EL USUARIO
# ============================================
st.markdown("---")
st.subheader("Interpretación del resultado final")

explicacion = """
A continuación se presenta una interpretación clara del análisis realizado entre los datos 
reales obtenidos por FlexSim y los resultados generados por nuestro Optimizer (Simplex):

1. **Producción Real (FlexSim):**  
   Representa el comportamiento real del sistema productivo bajo las condiciones actuales de la empresa.  
   Estos valores muestran cuántas unidades se lograron producir considerando restricciones reales 
   como tiempos, capacidad operativa, cuellos de botella y variabilidad del sistema.

2. **Producción Óptima (Optimizer):**  
   Corresponde a la mejor combinación posible de producción calculada mediante programación lineal, 
   asumiendo que las restricciones de materiales y demanda se cumplen, pero sin considerar 
   variabilidad, tiempos muertos o ineficiencia operativa.  
   Este valor representa el *máximo potencial teórico* de la operación.

3. **Diferencia (Óptimo – Real):**  
   • Si la diferencia es **positiva**, el sistema real está produciendo menos de lo que podría producir  
     en condiciones óptimas.  
   • Si la diferencia es **negativa**, la operación real está produciendo por encima del modelo teórico,  
     indicando estrategias adicionales no modeladas (turnos extra, buffers, paralelismo, etc.).  
   • Si la diferencia es cercana a cero, significa que el sistema real está trabajando muy ajustado al óptimo.

---

### ¿Qué significa esto para la empresa?

- El Optimizer muestra lo que la empresa **debería producir idealmente** para maximizar la ganancia.  
- FlexSim muestra lo que la empresa **realmente produce**, reflejando limitaciones físicas y operativas.  
- La comparativa evidencia si la empresa está **aprovechando su potencial**, si existen **ineficiencias**, 
  o si el sistema real supera las expectativas del modelo teórico.

---

En conclusión, esta comparación permite identificar oportunidades de mejora, validar procesos, 
encontrar cuellos de botella y evaluar qué tan eficiente es la operación actual frente a su máximo potencial.
"""

st.info(explicacion)
