import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre
from scipy.optimize import least_squares, lsq_linear, curve_fit
from scipy.linalg import cholesky, solve_triangular


def true_function(x, xc, gamma, c):
    """f(x) = (xc - x)^(-gamma) + c"""
    return (xc - x) ** (-gamma) + c


def true_first_derivative(x, xc, gamma):
    """f'(x) = gamma * (xc - x)^(-(gamma + 1))"""
    return gamma * (xc - x) ** (-(gamma + 1))


def true_second_derivative(x, xc, gamma):
    """f''(x) = gamma * (gamma + 1) * (xc - x)^(-(gamma + 2))"""
    return gamma * (gamma + 1) * (xc - x) ** (-(gamma + 2))


def test_hybrid_pade_nlls():
    # --- 1. Define true parameters and generate synthetic data ---
    xc_true = 5.0
    gamma_true = 2.1
    c_true = 2.0

    x0 = 4.2
    x_end = 4.8
    M = 50
    x_m = np.linspace(x0, x_end, M)
    dx = x_m[1] - x_m[0]

    noise_std = 0.5  # Increased noise level
    print(f"noise_std={noise_std}")

    # Generate noisy data
    np.random.seed(42)
    F_true = true_function(x_m, xc_true, gamma_true, c_true)
    F_noisy = F_true + np.random.normal(0, noise_std, size=M)

    # --- 2. Construct Differenced Data and Whiten ---
    d = F_noisy[1:] - F_noisy[:-1]

    # Covariance matrix V for differenced i.i.d noise (MA(1) process)
    V = np.diag(np.full(M - 1, 2.0)) + \
        np.diag(np.full(M - 2, -1.0), k=1) + \
        np.diag(np.full(M - 2, -1.0), k=-1)

    L = cholesky(V, lower=True)
    d_tilde = solve_triangular(L, d, lower=True)

    # --- 3. VARIABLE PROJECTION (Golub-Pereyra) ---
    n = 10
    t_initial, _ = roots_laguerre(n)
    m_diff_indices = np.arange(M - 1)

    # Tikhonov regularization parameter to prevent ill-conditioning in lsq_linear
    lambda_reg = 1e-6

    def varpro_residual(t_params):
        G_diff = np.zeros((M - 1, n))
        for j in range(n):
            G_diff[:, j] = np.exp(t_params[j] * m_diff_indices * dx) * (np.exp(t_params[j] * dx) - 1.0)

        G_tilde = solve_triangular(L, G_diff, lower=True)

        # Augment matrices for Tikhonov regularization
        G_aug = np.vstack((G_tilde, lambda_reg * np.eye(n)))
        d_aug = np.concatenate((d_tilde, np.zeros(n)))

        res_linear = lsq_linear(G_aug, d_aug, bounds=(0.0, np.inf))
        a_opt = res_linear.x
        return d_tilde - G_tilde @ a_opt

    # Changed lower bound from 0.0 to 1e-3 to prevent zero-columns
    res_nonlin = least_squares(varpro_residual, t_initial, bounds=(1e-3, 200.0), method='trf')
    t_opt = res_nonlin.x

    # --- 4. Extract Final Linear Parameters ---
    G_diff_opt = np.zeros((M - 1, n))
    for j in range(n):
        G_diff_opt[:, j] = np.exp(t_opt[j] * m_diff_indices * dx) * (np.exp(t_opt[j] * dx) - 1.0)

    G_tilde_opt = solve_triangular(L, G_diff_opt, lower=True)

    # Apply the same regularization for the final extraction
    G_aug_opt = np.vstack((G_tilde_opt, lambda_reg * np.eye(n)))
    d_aug_opt = np.concatenate((d_tilde, np.zeros(n)))

    a_opt = lsq_linear(G_aug_opt, d_aug_opt, bounds=(0.0, np.inf)).x
    w_tilde = a_opt * np.exp(-t_opt * x0)

    # --- 5. Compute Analytical Derivatives f' and f'' ---
    F_prime_fit = np.zeros(M)
    F_double_prime_fit = np.zeros(M)

    for j in range(n):
        F_prime_fit += w_tilde[j] * t_opt[j] * np.exp(t_opt[j] * x_m)
        F_double_prime_fit += w_tilde[j] * (t_opt[j] ** 2) * np.exp(t_opt[j] * x_m)

    # --- 6. Infer initial xc and gamma using d-log Padé (Ratio f'/f'') ---
    ratio_fit = F_prime_fit / F_double_prime_fit

    num_points_fit = 15
    x_m_reg = x_m[-num_points_fit:]
    ratio_reg = ratio_fit[-num_points_fit:]

    m_line, b_line = np.polyfit(x_m_reg, ratio_reg, 1)

    # Ratio = (xc - x) / (gamma + 1)
    gamma_pade = (-1.0 / m_line) - 1.0
    xc_pade = -b_line / m_line

    # Estimate initial c using the first data point
    # Use absolute value to prevent complex numbers if gamma_pade is weird
    c_pade = F_noisy[0] - np.abs(xc_pade - x_m[0])**(-gamma_pade)

    print("-" * 50)
    print("Initial Guesses from d-log Padé:")
    print(f"xc_pade:    {xc_pade:.4f}")
    print(f"gamma_pade: {gamma_pade:.4f}")
    print(f"c_pade:     {c_pade:.4f}")

    # --- 7. Direct 3-Parameter Non-Linear Least Squares (NLLS) ---
    def nlls_model(x, xc_val, gamma_val, c_val):
        return (xc_val - x)**(-gamma_val) + c_val

    p0 = [xc_pade, gamma_pade, c_pade]

    # Set bounds: xc must be strictly greater than the last data point to avoid complex numbers/singularities.
    bounds = ([x_m[-1] + 1e-5, 0.0, -np.inf], [np.inf, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(nlls_model, x_m, F_noisy, p0=p0, bounds=bounds)
        xc_nlls, gamma_nlls, c_nlls = popt

        print("-" * 50)
        print("Final Inferred Parameters (Direct 3-Parameter NLLS):")
        print(f"True xc:    {xc_true:.4f}  |  NLLS xc:    {xc_nlls:.4f}")
        print(f"True gamma: {gamma_true:.4f}  |  NLLS gamma: {gamma_nlls:.4f}")
        print(f"True c:     {c_true:.4f}  |  NLLS c:     {c_nlls:.4f}")
        print("-" * 50)

        F_nlls_fit = nlls_model(x_m, xc_nlls, gamma_nlls, c_nlls)
    except Exception as e:
        print(f"NLLS Fit failed: {e}")
        xc_nlls, gamma_nlls, c_nlls = xc_pade, gamma_pade, c_pade
        F_nlls_fit = np.zeros_like(x_m)

    # --- 8. Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: d-log Padé Ratio Fit
    ax1.plot(x_m, ratio_fit, 'mo', label='All Ratio Data', markersize=5, alpha=0.4)
    ax1.plot(x_m_reg, ratio_reg, 'co', label=f'Regression Pts (Last {num_points_fit})', markersize=7)
    ax1.plot(x_m, m_line * x_m + b_line, 'k-', label='Linear Regression', linewidth=2)
    ax1.set_title(r"Step 1: d-log Padé Ratio $\frac{f'(x)}{f''(x)}$")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Final NLLS Fit vs Noisy Data
    ax2.plot(x_m, F_noisy, 'ko', label='Noisy Data', markersize=4, alpha=0.6)
    ax2.plot(x_m, F_true, 'b-', label='True Function', linewidth=2, alpha=0.7)
    if np.any(F_nlls_fit):
        ax2.plot(x_m, F_nlls_fit, 'r--', label='Direct 3-Param NLLS Fit', linewidth=2)
    ax2.set_title("Step 2: Direct 3-Parameter NLLS Fit")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = (rf'Final NLLS Parameters:' '\n'
               rf'  $x_c$ = {xc_nlls:.4f} (True: {xc_true})' '\n'
               rf'  $\gamma$ = {gamma_nlls:.4f} (True: {gamma_true})' '\n'
               rf'  $c$ = {c_nlls:.4f} (True: {c_true})')
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("vp_pade_nlls.png")
    plt.close()


if __name__ == "__main__":
    test_hybrid_pade_nlls()