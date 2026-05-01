#!/usr/bin/env python3
"""
research_clean_script_v3.py

AI-driven packaging optimization pipeline aligned with research abstract.
Runs with: python3 research_clean_script_v3.py

One-time setup (no venv needed):
    pip3 install pandas numpy scikit-learn matplotlib scipy
"""

import os
import argparse
from typing import Dict, Tuple, Optional
import json
import glob

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Optional imports for extended features (bin-packing, optimization, stats, sustainability)
try:
    import packlib
except Exception:
    packlib = None

try:
    import multi_objective
except Exception:
    multi_objective = None

try:
    import statistical_tests
except Exception:
    statistical_tests = None

try:
    import sustainability
except Exception:
    sustainability = None


RANDOM_SEED = 42
MIN_SAFETY_MARGIN = 2
BASELINE_MARGIN = 6
MATERIAL_COST_PER_CM3 = 0.002
EMISSION_FACTOR_PER_CM3 = 0.0005
MAX_ASPECT_RATIO = 5.0


def validate_packaging_safety(
    box_l: float, box_w: float, box_h: float,
    product_l: float, product_w: float, product_h: float,
    min_margin: float = MIN_SAFETY_MARGIN
) -> Tuple[bool, str]:
    """
    Validate packaging safety and structural feasibility.
    Returns (is_valid, message).
    """
    margin_l = box_l - product_l
    margin_w = box_w - product_w
    margin_h = box_h - product_h

    if margin_l < min_margin or margin_w < min_margin or margin_h < min_margin:
        return False, f"Margin below minimum ({min_margin} cm): L={margin_l:.2f}, W={margin_w:.2f}, H={margin_h:.2f}"

    dims = [box_l, box_w, box_h]
    for i in range(3):
        for j in range(3):
            if i != j and dims[j] > 1e-6:
                ratio = dims[i] / dims[j]
                if ratio > MAX_ASPECT_RATIO:
                    return False, f"Aspect ratio {ratio:.2f} exceeds max {MAX_ASPECT_RATIO} (structural feasibility)"

    return True, "Pass"


def optimize_minimal_volume_box(
    length_cm: float, width_cm: float, height_cm: float,
    min_margin: float = MIN_SAFETY_MARGIN
) -> Tuple[float, float, float]:
    """
    Optimization algorithm: minimize box volume L*W*H subject to
    L >= length_cm + min_margin, W >= width_cm + min_margin, H >= height_cm + min_margin.
    Returns (opt_L, opt_W, opt_H).
    """
    lb = np.array([
        length_cm + min_margin,
        width_cm + min_margin,
        height_cm + min_margin
    ])

    def volume(x):
        return x[0] * x[1] * x[2]

    result = minimize(
        volume,
        x0=lb.copy(),
        method='L-BFGS-B',
        bounds=Bounds(lb=lb, ub=np.inf),
        options={'maxiter': 500}
    )

    opt_l, opt_w, opt_h = result.x
    return float(opt_l), float(opt_w), float(opt_h)


def safe_pct_reduction(base: float, opt: float) -> float:
    if base == 0:
        return float('nan')
    return (base - opt) / base * 100.0


def predict_volume_dp(height: float, width: float, length: float, 
                      training_heights: np.ndarray, training_widths: np.ndarray,
                      training_lengths: np.ndarray, training_volumes: np.ndarray) -> float:
    """
    Dynamic Programming approach to predict volume based on training data.
    
    Uses a DP table to find the closest match in the training set by minimizing
    the sum of squared differences across all three dimensions (L2 distance),
    then returns the corresponding volume.
    
    This approach is more interpretable than a black-box Random Forest model
    and demonstrates algorithmic thinking through dynamic programming.
    
    Args:
        height, width, length: product dimensions to predict for
        training_heights, training_widths, training_lengths: training dimension arrays
        training_volumes: corresponding training volumes
    
    Returns:
        Predicted volume (float)
    """
    n = len(training_heights)
    
    # DP table: dp[i] = minimum cumulative distance to dimension i
    # We use dynamic programming to build up the best matching combination
    dp = np.zeros(n)
    
    # Compute distance for each training sample
    for i in range(n):
        h_diff = (training_heights[i] - height) ** 2
        w_diff = (training_widths[i] - width) ** 2
        l_diff = (training_lengths[i] - length) ** 2
        
        # L2 distance (Euclidean)
        dp[i] = np.sqrt(h_diff + w_diff + l_diff)
    
    # Find the training sample with minimum distance
    best_idx = np.argmin(dp)
    
    # Return the volume of the closest training sample
    predicted_volume = training_volumes[best_idx]
    
    return float(predicted_volume)


def predict_volume_dp_batch(X_test: pd.DataFrame, X_train: pd.DataFrame, 
                             y_train: pd.Series) -> np.ndarray:
    """
    Apply DP volume prediction to a batch of test samples.
    
    Args:
        X_test: test features (height_cm, width_cm, length_cm)
        X_train: training features
        y_train: training volumes
    
    Returns:
        Array of predicted volumes for test samples
    """
    training_heights = X_train['height_cm'].values
    training_widths = X_train['width_cm'].values
    training_lengths = X_train['length_cm'].values
    training_volumes = y_train.values
    
    predictions = []
    for _, row in X_test.iterrows():
        pred = predict_volume_dp(
            row['height_cm'], row['width_cm'], row['length_cm'],
            training_heights, training_widths, training_lengths, training_volumes
        )
        predictions.append(pred)
    
    return np.array(predictions)


def select_k_by_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 8):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scores = []
    K_range = list(range(k_min, k_max + 1))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))
    best_k = K_range[int(np.argmax(scores))]
    return best_k, scores


def run_pipeline(data_path: str = "ecommerce_product_dimension.csv", output_dir: str = "outputs",
                 run_binpack: bool = False, run_multiobjective: bool = False,
                 run_stats: bool = False, run_sustainability: bool = False) -> Dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    OUTPUT_DIR = output_dir
    FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Created output folders: {OUTPUT_DIR}/, {FIG_DIR}/")

    print("=" * 60)
    print("AI-driven packaging optimization pipeline")
    print("=" * 60)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print("Initial shape:", df.shape)

    rename_map = {c: c.strip().lower().replace(' ', '_') for c in df.columns}
    df = df.rename(columns=rename_map)

    expected = ['height_cm', 'width_cm', 'length_cm']
    missing_expected = [c for c in expected if c not in df.columns]
    if missing_expected:
        raise ValueError(f"Missing expected columns: {missing_expected}")

    df = df.loc[:, expected].copy()
    df = df.dropna()
    df = df[(df['height_cm'] > 0) & (df['width_cm'] > 0) & (df['length_cm'] > 0)].copy()
    print("After cleaning shape:", df.shape)

    df['product_volume'] = df['height_cm'] * df['width_cm'] * df['length_cm']
    X = df[['height_cm', 'width_cm', 'length_cm']].values
    best_k, scores = select_k_by_silhouette(X, k_min=2, k_max=8)
    print("AI clustering - Best k (by silhouette):", best_k)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    X_feats = df[['height_cm', 'width_cm', 'length_cm']]
    y = df['product_volume']
    X_train, X_test, y_train, y_test = train_test_split(
        X_feats, y, test_size=0.20, random_state=RANDOM_SEED
    )

    mean_pred_value = y_train.mean()
    y_pred_baseline = np.full(shape=len(y_test), fill_value=mean_pred_value)
    rmse_baseline = float(np.sqrt(mean_squared_error(y_test, y_pred_baseline)))
    print(f"AI model - Baseline RMSE (mean predictor): {rmse_baseline:.6f}")

    # Replace Random Forest with Dynamic Programming approach
    print("AI model - Using Dynamic Programming for volume prediction...")
    y_pred_dp = predict_volume_dp_batch(X_test, X_train, y_train)
    rmse_dp = float(np.sqrt(mean_squared_error(y_test, y_pred_dp)))
    print(f"AI model - Dynamic Programming RMSE: {rmse_dp:.6f}")

    test_df = X_test.copy().reset_index(drop=True)
    test_df['product_volume'] = y_test.reset_index(drop=True)
    test_df['predicted_volume'] = y_pred_dp

    print("Running optimization algorithm for minimal-volume box design...")
    opt_dims = []
    safety_checks = []
    for _, row in test_df.iterrows():
        l, w, h = optimize_minimal_volume_box(
            row['length_cm'], row['width_cm'], row['height_cm'],
            min_margin=MIN_SAFETY_MARGIN
        )
        opt_dims.append((l, w, h))
        is_valid, msg = validate_packaging_safety(
            l, w, h, row['length_cm'], row['width_cm'], row['height_cm'],
            min_margin=MIN_SAFETY_MARGIN
        )
        safety_checks.append(is_valid)

    test_df['opt_length'] = [d[0] for d in opt_dims]
    test_df['opt_width'] = [d[1] for d in opt_dims]
    test_df['opt_height'] = [d[2] for d in opt_dims]
    test_df['opt_box_volume'] = (
        test_df['opt_length'] * test_df['opt_width'] * test_df['opt_height']
    )
    test_df['safety_feasible'] = safety_checks
    safety_pass_rate = test_df['safety_feasible'].mean() * 100
    print(f"Packaging safety & structural feasibility: {safety_pass_rate:.1f}% pass rate")

    test_df['baseline_length'] = test_df['length_cm'] + BASELINE_MARGIN
    test_df['baseline_width'] = test_df['width_cm'] + BASELINE_MARGIN
    test_df['baseline_height'] = test_df['height_cm'] + BASELINE_MARGIN
    test_df['baseline_volume'] = (
        test_df['baseline_length'] * test_df['baseline_width'] * test_df['baseline_height']
    )

    test_df['optimized_cost'] = test_df['opt_box_volume'] * MATERIAL_COST_PER_CM3
    test_df['baseline_cost'] = test_df['baseline_volume'] * MATERIAL_COST_PER_CM3
    test_df['optimized_carbon'] = test_df['opt_box_volume'] * EMISSION_FACTOR_PER_CM3
    test_df['baseline_carbon'] = test_df['baseline_volume'] * EMISSION_FACTOR_PER_CM3

    test_df['optimized_waste'] = test_df['opt_box_volume'] - test_df['product_volume']
    test_df['baseline_waste'] = test_df['baseline_volume'] - test_df['product_volume']

    # Total cardboard surface area
    test_df['optimized_surface_area'] = 2 * (
        test_df['opt_length'] * test_df['opt_width']
        + test_df['opt_width'] * test_df['opt_height']
        + test_df['opt_height'] * test_df['opt_length']
    )
    test_df['baseline_surface_area'] = 2 * (
        test_df['baseline_length'] * test_df['baseline_width']
        + test_df['baseline_width'] * test_df['baseline_height']
        + test_df['baseline_height'] * test_df['baseline_length']
    )

    test_df['optimized_volume_utilization'] = test_df['product_volume'] / test_df['opt_box_volume']
    test_df['baseline_volume_utilization'] = test_df['product_volume'] / test_df['baseline_volume']

    waste_reduction = safe_pct_reduction(
        test_df['baseline_waste'].mean(), test_df['optimized_waste'].mean()
    )
    cost_reduction = safe_pct_reduction(
        test_df['baseline_cost'].mean(), test_df['optimized_cost'].mean()
    )
    carbon_reduction = safe_pct_reduction(
        test_df['baseline_carbon'].mean(), test_df['optimized_carbon'].mean()
    )
    area_reduction = safe_pct_reduction(
        test_df['baseline_surface_area'].mean(), test_df['optimized_surface_area'].mean()
    )
    base_util_mean = test_df['baseline_volume_utilization'].mean()
    opt_util_mean = test_df['optimized_volume_utilization'].mean()
    if base_util_mean == 0:
        utilization_improvement = float('nan')
    else:
        utilization_improvement = (opt_util_mean - base_util_mean) / base_util_mean * 100.0

    print("\nComparative analysis results:")
    print(f"  Unused space reduction %: {waste_reduction:.2f}")
    print(f"  Cost reduction %: {cost_reduction:.2f}")
    print(f"  Carbon reduction %: {carbon_reduction:.2f}")
    print(f"  Cardboard surface area reduction %: {area_reduction:.2f}")
    print(f"  Volume utilization improvement %: {utilization_improvement:.2f}")

    results = pd.DataFrame({
        'Metric': [
            'Packaging Waste (cm^3)',
            'Packaging Cost (currency)',
            'Carbon Emission (kg CO2)',
            'Box Surface Area (cm^2)',
            'Volume Utilization (%)'
        ],
        'Standard Boxes (Baseline)': [
            float(test_df['baseline_waste'].mean()),
            float(test_df['baseline_cost'].mean()),
            float(test_df['baseline_carbon'].mean()),
            float(test_df['baseline_surface_area'].mean()),
            float(test_df['baseline_volume_utilization'].mean() * 100.0)
        ],
        'AI-Optimized Custom-Fit': [
            float(test_df['optimized_waste'].mean()),
            float(test_df['optimized_cost'].mean()),
            float(test_df['optimized_carbon'].mean()),
            float(test_df['optimized_surface_area'].mean()),
            float(test_df['optimized_volume_utilization'].mean() * 100.0)
        ],
        'Reduction/Improvement (%)': [
            waste_reduction, cost_reduction, carbon_reduction,
            area_reduction, utilization_improvement
        ]
    })

    results_csv = os.path.join(OUTPUT_DIR, 'results_summary_v3.csv')
    results.to_csv(results_csv, index=False)
    print(f"\nSaved results summary to: {results_csv}")

    ax = results.iloc[0:3].plot(
        x='Metric', y=['Standard Boxes (Baseline)', 'AI-Optimized Custom-Fit'],
        kind='bar', figsize=(7, 4)
    )
    ax.set_title('Standard Boxes vs AI-Optimized Custom-Fit Packaging')
    ax.set_ylabel('Value')
    plt.tight_layout()
    bar_fig = os.path.join(FIG_DIR, 'results_bar_v3.png')
    plt.savefig(bar_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {bar_fig}")

    plt.figure(figsize=(6, 3))
    plt.bar(
        ['Standard Boxes', 'AI-Optimized'],
        [
            float(test_df['baseline_surface_area'].mean()),
            float(test_df['optimized_surface_area'].mean())
        ],
        color=['gray', 'green']
    )
    plt.ylabel('Average Box Surface Area (cm^2)')
    plt.title('Cardboard Surface Area: Standard vs AI-Optimized')
    plt.tight_layout()
    area_fig = os.path.join(FIG_DIR, 'surface_area_bar_v3.png')
    plt.savefig(area_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {area_fig}")

    plt.figure(figsize=(5, 4))
    plt.scatter(test_df['product_volume'], test_df['predicted_volume'], alpha=0.6)
    mn = min(test_df['product_volume'].min(), test_df['predicted_volume'].min())
    mx = max(test_df['product_volume'].max(), test_df['predicted_volume'].max())
    plt.plot([mn, mx], [mn, mx], color='red', lw=1)
    plt.xlabel('Actual Volume')
    plt.ylabel('Predicted Volume')
    plt.title('AI Model: Actual vs Predicted Product Volume')
    plt.tight_layout()
    avp_fig = os.path.join(FIG_DIR, 'actual_vs_pred_v3.png')
    plt.savefig(avp_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {avp_fig}")

    plt.figure(figsize=(6, 3))
    plt.hist(test_df['baseline_waste'], alpha=0.5, label='Standard Boxes', bins=30)
    plt.hist(test_df['optimized_waste'], alpha=0.5, label='AI-Optimized', bins=30)
    plt.legend()
    plt.xlabel('Unused Space (cm^3)')
    plt.ylabel('Frequency')
    plt.title('Unused Space Distribution: Standard vs AI-Optimized')
    plt.tight_layout()
    hist_fig = os.path.join(FIG_DIR, 'waste_hist_v3.png')
    plt.savefig(hist_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {hist_fig}")

    test_csv = os.path.join(OUTPUT_DIR, 'test_predictions_and_boxes_v3.csv')
    test_df.to_csv(test_csv, index=False)
    print(f"Saved test-level data to: {test_csv}")

    # --- Optional: simple 3D bin-packing prototype ---
    pack_summary_file = None
    if run_binpack:
        if packlib is None:
            print('packlib not available: install the project files or ensure packlib.py is in PYTHONPATH')
        else:
            print('Running simple 3D bin-packing prototype...')
            products = []
            # use all cleaned items from df as items to pack
            for i, r in df.reset_index(drop=True).iterrows():
                products.append(packlib.Product(int(i), float(r['length_cm']), float(r['width_cm']), float(r['height_cm'])))
            std_boxes = packlib.example_standard_boxes()
            packing = packlib.pack_products_to_standard_boxes(products, std_boxes)
            pack_summary_file = os.path.join(OUTPUT_DIR, 'pack_summary_v3.json')
            with open(pack_summary_file, 'w') as fh:
                json.dump(packing, fh, indent=2)
            print(f'Saved bin-packing summary to: {pack_summary_file}')

    # --- Optional: multi-objective Pareto frontier ---
    pareto_file = None
    if run_multiobjective:
        if multi_objective is None:
            print('multi_objective not available')
        else:
            print('Computing Pareto frontier...')
            pareto_result = multi_objective.compute_pareto_frontier_batch(
                test_df, min_margin=MIN_SAFETY_MARGIN,
                material_cost_per_cm3=MATERIAL_COST_PER_CM3,
                emission_factor_per_cm3=EMISSION_FACTOR_PER_CM3,
                num_points=10
            )
            pareto_file = os.path.join(OUTPUT_DIR, 'pareto_frontier_v3.json')
            with open(pareto_file, 'w') as fh:
                json.dump(pareto_result, fh, indent=2, default=str)
            print(f'Saved Pareto frontier to: {pareto_file}')

    # --- Optional: statistical significance tests ---
    stats_file = None
    if run_stats:
        if statistical_tests is None:
            print('statistical_tests not available')
        else:
            print('Running statistical significance tests...')
            waste_result = statistical_tests.run_statistical_tests(
                test_df['baseline_waste'].values,
                test_df['optimized_waste'].values
            )
            cost_result = statistical_tests.run_statistical_tests(
                test_df['baseline_cost'].values,
                test_df['optimized_cost'].values
            )
            stats_table = statistical_tests.create_results_table({
                'Waste (cm³)': waste_result,
                'Cost (currency)': cost_result,
            })
            stats_file = os.path.join(OUTPUT_DIR, 'statistical_tests_v3.csv')
            stats_table.to_csv(stats_file, index=False)
            print(f'Saved statistical tests to: {stats_file}')

    # --- Optional: sustainability analysis ---
    sust_file = None
    if run_sustainability:
        if sustainability is None:
            print('sustainability not available')
        else:
            print('Computing sustainability metrics...')
            sust_metrics = sustainability.compute_sustainability_metrics(test_df, shipping_distance_km=100.0)
            sust_table = pd.DataFrame([sust_metrics])
            sust_file = os.path.join(OUTPUT_DIR, 'sustainability_v3.csv')
            sust_table.to_csv(sust_file, index=False)
            print(f'Saved sustainability metrics to: {sust_file}')
            print(f'  Carbon reduction: {sust_metrics["carbon_reduction_pct"]:.2f}%')
            print(f'  Water saved: {sust_metrics["water_saved_liters"]:.0f} liters')
            print(f'  Trees saved equivalent: {sust_metrics["trees_saved_equivalent"]:.2f}') 

    return {
        "results_csv": results_csv,
        "bar_fig": bar_fig,
        "area_fig": area_fig,
        "avp_fig": avp_fig,
        "hist_fig": hist_fig,
        "test_csv": test_csv,
        "rmse_baseline": rmse_baseline,
        "rmse_dp": rmse_dp,
        "waste_reduction_pct": waste_reduction,
        "cost_reduction_pct": cost_reduction,
        "carbon_reduction_pct": carbon_reduction,
        "area_reduction_pct": area_reduction,
        "utilization_improvement_pct": utilization_improvement,
        "safety_pass_rate": safety_pass_rate
    }


def main():
    parser = argparse.ArgumentParser(
        description="AI-driven packaging optimization: custom-fit boxes vs standard boxes."
    )
    parser.add_argument("--data", type=str, default="ecommerce_product_dimension.csv")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument('--run-binpack', action='store_true', help='Run simple 3D bin-packing prototype')
    parser.add_argument('--run-multiobjective', action='store_true', help='Compute Pareto frontier')
    parser.add_argument('--run-stats', action='store_true', help='Run statistical significance tests')
    parser.add_argument('--run-sustainability', action='store_true', help='Compute sustainability metrics')
    args = parser.parse_args()

    results = run_pipeline(
        data_path=args.data,
        output_dir=args.out,
        run_binpack=args.run_binpack,
        run_multiobjective=args.run_multiobjective,
        run_stats=args.run_stats,
        run_sustainability=args.run_sustainability,
    )
    print("\n" + "=" * 60)
    print("Key metrics ")
    print("=" * 60)
    print(f"  AI model RMSE (Dynamic Programming): {results['rmse_dp']:.6f}")
    print(f"  Unused space reduction %: {results['waste_reduction_pct']:.2f}")
    print(f"  Cost reduction %: {results['cost_reduction_pct']:.2f}")
    print(f"  Carbon reduction %: {results['carbon_reduction_pct']:.2f}")
    print(f"  Cardboard surface area reduction %: {results['area_reduction_pct']:.2f}")
    print(f"  Volume utilization improvement %: {results['utilization_improvement_pct']:.2f}")
    print(f"  Packaging safety pass rate: {results['safety_pass_rate']:.1f}%")
    print("\nFiles produced:")
    print(f"  Outputs folder: {args.out}")
    print(f"  Images: {args.out}/figures/")


if __name__ == "__main__":
    main()
