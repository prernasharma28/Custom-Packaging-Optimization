"""multi_objective.py

Multi-objective packaging optimization using Pareto frontier.
Simultaneously optimizes: minimize waste, minimize cost, minimize carbon.
Returns a set of non-dominated solutions (Pareto frontier).
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import minimize, Bounds


def pareto_frontier(objectives: List[Tuple[float, ...]]) -> List[int]:
    """
    Compute Pareto frontier from a set of objectives.
    Each objective is a tuple (waste, cost, carbon).
    Returns indices of non-dominated solutions.
    
    A solution is non-dominated if no other solution is better in all objectives.
    (Assuming all objectives are to be minimized.)
    """
    objectives = np.array(objectives)
    n = len(objectives)
    is_dominated = [False] * n
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if j dominates i (j is better in all objectives)
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                is_dominated[i] = True
                break
    
    return [i for i in range(n) if not is_dominated[i]]


def optimize_pareto_box(
    length_cm: float, width_cm: float, height_cm: float,
    min_margin: float = 2.0,
    material_cost_per_cm3: float = 0.002,
    emission_factor_per_cm3: float = 0.0005,
    product_volume: float = None,
    num_points: int = 10
) -> List[Dict]:
    """
    Generate Pareto-optimal boxes by trading off waste, cost, and carbon.
    
    Vary a trade-off parameter lambda in [0, 1]:
    - lambda=0: minimize waste (volume)
    - lambda=0.5: balanced
    - lambda=1: minimize cost+carbon (prioritize material efficiency)
    
    Returns: list of dicts with keys: 'l', 'w', 'h', 'volume', 'cost', 'carbon', 'waste', 'lambda'
    """
    if product_volume is None:
        product_volume = length_cm * width_cm * height_cm
    
    lb = np.array([
        length_cm + min_margin,
        width_cm + min_margin,
        height_cm + min_margin
    ])
    
    solutions = []
    
    for lam_idx in range(num_points):
        lam = lam_idx / (num_points - 1) if num_points > 1 else 0.5
        
        def objective(x):
            vol = x[0] * x[1] * x[2]
            waste = vol - product_volume
            cost = vol * material_cost_per_cm3
            carbon = vol * emission_factor_per_cm3
            # Weighted sum: minimize waste and (cost + carbon)
            return (1 - lam) * waste + lam * (cost + carbon)
        
        result = minimize(
            objective,
            x0=lb.copy(),
            method='L-BFGS-B',
            bounds=Bounds(lb=lb, ub=np.inf),
            options={'maxiter': 500}
        )
        
        opt_l, opt_w, opt_h = result.x
        opt_vol = opt_l * opt_w * opt_h
        opt_cost = opt_vol * material_cost_per_cm3
        opt_carbon = opt_vol * emission_factor_per_cm3
        opt_waste = opt_vol - product_volume
        
        solutions.append({
            'l': float(opt_l),
            'w': float(opt_w),
            'h': float(opt_h),
            'volume': float(opt_vol),
            'cost': float(opt_cost),
            'carbon': float(opt_carbon),
            'waste': float(opt_waste),
            'lambda': float(lam)
        })
    
    return solutions


def compute_pareto_frontier_batch(
    test_df,
    min_margin: float = 2.0,
    material_cost_per_cm3: float = 0.002,
    emission_factor_per_cm3: float = 0.0005,
    num_points: int = 10
) -> Dict:
    """
    Compute Pareto frontier for all items in test_df.
    
    Returns: {
        'frontiers': [list of solutions for each item],
        'avg_waste': mean waste across frontier,
        'avg_cost': mean cost across frontier,
        'avg_carbon': mean carbon across frontier,
    }
    """
    all_frontiers = []
    
    for idx, row in test_df.iterrows():
        solutions = optimize_pareto_box(
            row['length_cm'], row['width_cm'], row['height_cm'],
            min_margin=min_margin,
            material_cost_per_cm3=material_cost_per_cm3,
            emission_factor_per_cm3=emission_factor_per_cm3,
            product_volume=row['product_volume'],
            num_points=num_points
        )
        all_frontiers.append(solutions)
    
    # Aggregate metrics
    all_waste = [s['waste'] for frontier in all_frontiers for s in frontier]
    all_cost = [s['cost'] for frontier in all_frontiers for s in frontier]
    all_carbon = [s['carbon'] for frontier in all_frontiers for s in frontier]
    
    return {
        'frontiers': all_frontiers,
        'avg_waste': float(np.mean(all_waste)),
        'avg_cost': float(np.mean(all_cost)),
        'avg_carbon': float(np.mean(all_carbon)),
        'num_solutions': len(all_waste)
    }


if __name__ == '__main__':
    # Test
    sols = optimize_pareto_box(10, 8, 5, num_points=5)
    for s in sols:
        print(f"lambda={s['lambda']:.2f}: vol={s['volume']:.2f}, waste={s['waste']:.2f}, cost={s['cost']:.4f}, carbon={s['carbon']:.6f}")
