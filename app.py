import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px

st.set_page_config(page_title="Optimizador de Producción y Precios", layout="wide")

st.title("🧮 Optimizador de Producción y Precios (con visualización)")

# --- DATOS DE ENTRADA ---
st.sidebar.header("⚙️ Configuración")
load_example = st.sidebar.checkbox("Cargar datos de ejemplo")

if load_example:
    materials = pd.DataFrame({
        "Materia prima": ["Madera", "Pegante"],
        "Disponibilidad": [50, 50]
    })
    products = pd.DataFrame({
        "Producto": ["Silla", "Mesa"],
        "Ganancia": [40, 30],
        "Madera": [3, 2],
        "Pegante": [2, 1]
    })
else:
    st.sidebar.subheader("Materias primas")
    num_materials = st.sidebar.number_input("Número de materias primas", 1, 10, 2)
    mat_names = [st.sidebar.text_input(f"Materia prima {i+1}", f"Materia {i+1}") for i in range(num_materials)]
    mat_avail = [st.sidebar.number_input(f"Disponibilidad de {mat_names[i]}", 0.0, 10000.0, 50.0) for i in range(num_materials)]
    materials = pd.DataFrame({"Materia prima": mat_names, "Disponibilidad": mat_avail})

    st.sidebar.subheader("Productos")
    num_products = st.sidebar.number_input("Número de productos", 1, 10, 2)
    prod_names = [st.sidebar.text_input(f"Producto {i+1}", f"Producto {i+1}") for i in range(num_products)]
    prod_profit = [st.sidebar.number_input(f"Ganancia de {prod_names[i]}", 0.0, 10000.0, 10.0) for i in range(num_products)]

    usage = []
    for i in range(num_products):
        usage.append([st.sidebar.number_input(f"{prod_names[i]} usa de {m}", 0.0, 1000.0, 1.0)
                      for m in mat_names])

    df_usage = pd.DataFrame(usage, columns=mat_names)
    products = pd.concat([pd.DataFrame({"Producto": prod_names, "Ganancia": prod_profit}), df_usage], axis=1)

st.subheader("📋 Datos actuales")
st.write("**Materias primas:**")
st.dataframe(materials)
st.write("**Productos:**")
st.dataframe(products)

# --- OPTIMIZACIÓN ---
if st.button("🚀 Ejecutar Simplex"):
    try:
        profits = products["Ganancia"].to_numpy()
        A = products[materials["Materia prima"]].to_numpy().T
        b = materials["Disponibilidad"].to_numpy()

        res = linprog(c=-profits, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

        if res.success:
            st.success("✅ Optimización completada con éxito")

            results = pd.DataFrame({
                "Producto": products["Producto"],
                "Cantidad óptima": np.round(res.x, 2)
            })

            st.subheader("📦 Producción óptima:")
            st.dataframe(results)

            st.metric("💰 Ganancia máxima", f"${abs(res.fun):,.2f}")

            # --- VISUALIZACIONES ---
            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.bar(results, x="Producto", y="Cantidad óptima",
                              title="Producción óptima por producto",
                              color="Producto", text="Cantidad óptima")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                used = np.dot(A, res.x)
                mat_usage = pd.DataFrame({
                    "Materia prima": materials["Materia prima"],
                    "Usado": np.round(used, 2),
                    "Disponible": materials["Disponibilidad"]
                })
                fig2 = px.bar(mat_usage, x="Materia prima", y=["Usado", "Disponible"],
                              barmode="group", title="Uso de materias primas")
                st.plotly_chart(fig2, use_container_width=True)

            # --- Interpretación ---
            st.subheader("🧠 Interpretación automática")
            best_prod = results.loc[results["Cantidad óptima"].idxmax(), "Producto"]
            st.info(f"El producto **{best_prod}** tiene la mayor producción óptima. "
                    f"La ganancia total es **${abs(res.fun):,.2f}**.")

        else:
            st.error("❌ No se pudo encontrar una solución óptima.")

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
