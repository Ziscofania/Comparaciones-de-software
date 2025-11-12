# optimizer.py
import numpy as np
from scipy.optimize import linprog, minimize

def solve_quantity_lp(products, materials, prices):
    """
    Optimiza cantidades (x_i) con precios fijos usando programación lineal (Simplex).
    Maximize sum ( (price_i - unit_cost_i) * x_i )
    s.t. material constraints: sum(material_usage_ij * x_i) <= available_j
         demand caps: 0 <= x_i <= demand_i (a - b*p)
    """
    n = len(products)
    # Objective: maximize profit -> linprog minimizes, so minimize negative profit
    c = []
    bounds = []
    for i, prod in enumerate(products):
        profit_per_unit = prices[i] - prod["unit_cost"]
        c.append(-profit_per_unit)  # minimize negative => maximize profit
        # demand cap:
        a = prod.get("demand_a", 0)
        b = prod.get("demand_b", 0.0)
        demand_cap = max(0.0, a - b * prices[i])
        bounds.append((0, demand_cap))

    # Material constraints matrix
    mat_names = [m["name"] for m in materials]
    A = []
    b = []
    for mat in materials:
        row = []
        for prod in products:
            row.append(prod.get("materials", {}).get(mat["name"], 0.0))
        A.append(row)
        b.append(mat["available"])

    res = linprog(c=np.array(c), A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    if not res.success:
        return {"success": False, "message": res.message}
    x = res.x
    total_profit = sum((prices[i] - products[i]["unit_cost"]) * x[i] for i in range(n))
    return {"success": True, "quantities": x, "profit": total_profit, "message": res.message}

def profit_for_prices(prices, products, materials):
    """
    Given vector of prices, compute quantities q_i = max(0, a - b*p)
    and check material limits. If materials exceeded, penalize heavily.
    Returns negative profit (for minimization).
    """
    n = len(products)
    qs = np.zeros(n)
    for i, p in enumerate(prices):
        a = products[i].get("demand_a", 0.0)
        b = products[i].get("demand_b", 0.0)
        q = max(0.0, a - b * p)
        qs[i] = q

    # profit
    profit = sum((prices[i] - products[i]["unit_cost"]) * qs[i] for i in range(n))

    # material usage
    mat_names = [m["name"] for m in materials]
    penalties = 0.0
    for j, mat in enumerate(materials):
        use_j = 0.0
        for i, prod in enumerate(products):
            use_j += prod.get("materials", {}).get(mat["name"], 0.0) * qs[i]
        if use_j > mat["available"] + 1e-8:
            # penalty proportional to exceed
            penalties += 1e6 * (use_j - mat["available"])
    # We want to maximize profit -> minimize negative profit + penalties
    return -profit + penalties

def optimize_prices(products, materials, price_bounds=None, x0=None):
    """
    Optimize continuous prices (one price per product).
    Uses scipy.minimize on negative profit (with penalty for material constraint).
    price_bounds: list of (min, max) for each product
    """
    n = len(products)
    if price_bounds is None:
        price_bounds = [(prod.get("price_min", 0.0), prod.get("price_max", prod.get("demand_a", 1000))) for prod in products]
    if x0 is None:
        x0 = np.array([(b[0] + b[1]) / 2.0 for b in price_bounds])

    res = minimize(lambda p: profit_for_prices(p, products, materials),
                   x0, bounds=price_bounds, method='L-BFGS-B')
    if not res.success:
        return {"success": False, "message": res.message}
    prices_opt = res.x
    # compute q and profit
    qs = np.array([max(0.0, prod.get("demand_a", 0.0) - prod.get("demand_b", 0.0) * prices_opt[i])
                   for i, prod in enumerate(products)])
    total_profit = sum((prices_opt[i] - products[i]["unit_cost"]) * qs[i] for i in range(n))

    return {"success": True, "prices": prices_opt, "quantities": qs, "profit": total_profit, "message": res.message}
