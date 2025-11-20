# optimizer.py
import numpy as np
from scipy.optimize import linprog, minimize

"""
MÓDULO PROFESIONAL DE OPTIMIZACIÓN
----------------------------------
Produce:
1. Optimización de cantidades (LP - Simplex)
2. Optimización simultánea de precios y cantidades
3. Datos comparativos entre:
   - Producción actual (empresa real / FlexSim)
   - Producción óptima (modelo matemático)
   - Costos, ganancias, demanda, uso de materiales
4. Estructuras listas para Plotly (tu app.py solo grafica)
"""


# -------------------------------------------------------
# 1. OPTIMIZACIÓN DE CANTIDADES (Simplex)
# -------------------------------------------------------
def solve_quantity_lp(products, materials, prices):
    n = len(products)

    c = []
    bounds = []

    for i, prod in enumerate(products):
        profit_u = prices[i] - prod["unit_cost"]
        c.append(-profit_u)

        a = prod.get("demand_a", 0)
        b = prod.get("demand_b", 0)
        demand_cap = max(0.0, a - b * prices[i])

        bounds.append((0, demand_cap))

    A = []
    b_vec = []
    for mat in materials:
        row = []
        for prod in products:
            row.append(prod.get("materials", {}).get(mat["name"], 0))
        A.append(row)
        b_vec.append(mat["available"])

    res = linprog(c=np.array(c), A_ub=np.array(A), b_ub=np.array(b_vec),
                  bounds=bounds, method="highs")

    if not res.success:
        return {"success": False, "message": res.message}

    x = res.x
    total_profit = sum((prices[i] - products[i]["unit_cost"]) * x[i] for i in range(n))

    return {
        "success": True,
        "quantities": x,
        "profit": total_profit,
        "message": res.message,
        "material_usage": compute_material_usage(products, materials, x)
    }


# -------------------------------------------------------
# 2. FUNCIÓN DE GANANCIA → PARA OPTIMIZACIÓN DE PRECIOS
# -------------------------------------------------------
def profit_for_prices(prices, products, materials):
    n = len(products)
    qs = np.zeros(n)

    for i, p in enumerate(prices):
        a = products[i].get("demand_a", 0)
        b = products[i].get("demand_b", 0)
        qs[i] = max(0, a - b * p)

    profit = sum((prices[i] - products[i]["unit_cost"]) * qs[i] for i in range(n))

    penalties = 0
    for mat in materials:
        usage = 0
        for i, prod in enumerate(products):
            usage += prod.get("materials", {}).get(mat["name"], 0) * qs[i]

        if usage > mat["available"]:
            penalties += 1e6 * (usage - mat["available"])

    return -profit + penalties


# -------------------------------------------------------
# 3. OPTIMIZACIÓN DE PRECIOS
# -------------------------------------------------------
def optimize_prices(products, materials, price_bounds=None, x0=None):
    n = len(products)

    if price_bounds is None:
        price_bounds = []
        for prod in products:
            mn = prod.get("price_min", 1.0)
            mx = prod.get("price_max", prod.get("demand_a", 1000))
            price_bounds.append((mn, mx))

    if x0 is None:
        x0 = np.array([(mn + mx) / 2 for mn, mx in price_bounds])

    res = minimize(lambda p: profit_for_prices(p, products, materials),
                   x0, bounds=price_bounds, method="L-BFGS-B")

    if not res.success:
        return {"success": False, "message": res.message}

    prices_opt = res.x
    qs = np.array([
        max(0, prod.get("demand_a", 0) - prod.get("demand_b", 0) * prices_opt[i])
        for i, prod in enumerate(products)
    ])

    total_profit = sum((prices_opt[i] - products[i]["unit_cost"]) * qs[i] for i in range(n))

    return {
        "success": True,
        "prices": prices_opt,
        "quantities": qs,
        "profit": total_profit,
        "material_usage": compute_material_usage(products, materials, qs)
    }


# -------------------------------------------------------
# 4. COMPUTA USO DE MATERIALES
# -------------------------------------------------------
def compute_material_usage(products, materials, quantities):
    usage = {}
    for mat in materials:
        name = mat["name"]
        total_use = 0
        for i, prod in enumerate(products):
            total_use += prod.get("materials", {}).get(name, 0) * quantities[i]
        usage[name] = total_use
    return usage


# -------------------------------------------------------
# 5. GENERA INFORME COMPARATIVO
# -------------------------------------------------------
def comparative_report(real_data, simplex_result, price_opt_result):
    """
    real_data: dict con producción real (FlexSim o empresa)
        { "Producto1": cantidad_real, ... }

    simplex_result: resultado de solve_quantity_lp
    price_opt_result: resultado de optimize_prices
    """

    comps = []

    for i, prod in enumerate(real_data["product_names"]):
        comps.append({
            "product": prod,
            "real_production": real_data["quantities"][i],
            "simplex_opt": simplex_result["quantities"][i],
            "price_opt_q": price_opt_result["quantities"][i],
            "price_opt_p": price_opt_result["prices"][i],
        })

    return {
        "comparison": comps,
        "profits": {
            "real_profit": real_data["profit"],
            "simplex_profit": simplex_result["profit"],
            "price_opt_profit": price_opt_result["profit"]
        }
    }
def run_optimizer(input_data):
    """
    Ejecuta el algoritmo del optimizador y retorna un DataFrame
    compatible con la app.
    """
    # Aquí llamas a tu función real que hace la optimización.
    # Ejemplo (modifícalo según tu código):
    df_opt = optimize_production(input_data)

    return df_opt
