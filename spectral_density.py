import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre
from scipy.optimize import least_squares, lsq_linear
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


def test_hybrid_pade_spectral():
    # --- 1. Define true parameters and generate synthetic data ---
    xc_true = 5.0
    gamma_true = 2.1
    c_true = 2.0

    x0 = 4.2
    x_end = 4.8
    M = 50
    x_m = np.linspace(x0, x_end, M)
    dx = x_m[1] - x_m[0]

    noise_std = 0.01
    print(f"noise_std={noise_std}")
    # Generate noisy data
    np.random.seed(42)
    F_true = true_function(x_m, xc_true, gamma_true, c_true)
    F_noisy = F_true + np.random.normal(0, noise_std, size=M)

    F_prime_true = true_first_derivative(x_m, xc_true, gamma_true)
    F_double_prime_true = true_second_derivative(x_m, xc_true, gamma_true)

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

    def varpro_residual(t_params):
        G_diff = np.zeros((M - 1, n))
        for j in range(n):
            G_diff[:, j] = np.exp(t_params[j] * m_diff_indices * dx) * (np.exp(t_params[j] * dx) - 1.0)

        G_tilde = solve_triangular(L, G_diff, lower=True)
        res_linear = lsq_linear(G_tilde, d_tilde, bounds=(0.0, np.inf))
        a_opt = res_linear.x
        residual = d_tilde - G_tilde @ a_opt
        return residual

    res_nonlin = least_squares(varpro_residual, t_initial, bounds=(0.0, 200.0), method='trf')
    t_opt = res_nonlin.x

    # --- 4. Extract Final Linear Parameters ---
    G_diff_opt = np.zeros((M - 1, n))
    for j in range(n):
        G_diff_opt[:, j] = np.exp(t_opt[j] * m_diff_indices * dx) * (np.exp(t_opt[j] * dx) - 1.0)

    G_tilde_opt = solve_triangular(L, G_diff_opt, lower=True)
    a_opt = lsq_linear(G_tilde_opt, d_tilde, bounds=(0.0, np.inf)).x
    w_tilde = a_opt * np.exp(-t_opt * x0)

    # --- 5. SORT t, a, w ---
    sort_idx = np.argsort(t_opt)
    t_sorted = t_opt[sort_idx]
    a_sorted = a_opt[sort_idx]
    w_sorted = w_tilde[sort_idx]

    print(f"t_opt (sorted)={t_sorted}")
    print(f"a_opt (sorted)={a_sorted}")
    print(f"w_tilde (sorted)={w_sorted}")

    # --- 6. Compute Analytical Derivatives f' and f'' ---
    F_prime_fit = np.zeros(M)
    F_double_prime_fit = np.zeros(M)

    for j in range(n):
        F_prime_fit += w_sorted[j] * t_sorted[j] * np.exp(t_sorted[j] * x_m)
        F_double_prime_fit += w_sorted[j] * (t_sorted[j] ** 2) * np.exp(t_sorted[j] * x_m)

    # --- 7. Infer xc using d-log Padé (Ratio f'/f'') ---
    ratio_fit = F_prime_fit / F_double_prime_fit

    num_points_fit = 15
    x_m_reg = x_m[-num_points_fit:]
    ratio_reg = ratio_fit[-num_points_fit:]

    m_line, b_line = np.polyfit(x_m_reg, ratio_reg, 1)

    # From ratio = (xc - x) / (gamma + 1)
    # y = mx + b => m = -1/(gamma+1), b = xc/(gamma+1) => xc = -b/m
    xc_inferred = -b_line / m_line

    # --- 8. Infer gamma using Spectral Density Method ---
    # Filter out inactive poles (amplitudes driven to ~0 by VarPro bounds)
    active_mask = a_sorted > 1e-8
    t_active = t_sorted[active_mask]
    a_active = a_sorted[active_mask]

    if len(t_active) >= 2:
        # Model: log(a) + (xc - x0)*t = C + (gamma - 1)*log(t)
        Y = np.log(a_active) + (xc_inferred - x0) * t_active
        X = np.log(t_active)

        m_spec, b_spec = np.polyfit(X, Y, 1)
        gamma_inferred = m_spec + 1.0

        print("-" * 50)
        print("Inferred Critical Parameters (Hybrid Method):")
        print(f"True xc:    {xc_true:.4f}  |  Inferred xc (d-log Padé): {xc_inferred:.4f}")
        print(f"True gamma: {gamma_true:.4f}  |  Inferred gamma (Spectral): {gamma_inferred:.4f}")
        print("-" * 50)

        # --- 9. Plotting ---
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Ratio for xc
        ax1.plot(x_m, ratio_fit, 'mo', label='All Ratio Data', markersize=5, alpha=0.4)
        ax1.plot(x_m_reg, ratio_reg, 'co', label=f'Regression Pts (Last {num_points_fit})', markersize=7)
        ax1.plot(x_m, m_line * x_m + b_line, 'k-', label='Linear Regression', linewidth=2)
        ax1.set_title(r"d-log Padé: $\frac{f'(x)}{f''(x)}$ for $x_c$")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Spectral Density for gamma
        ax2.scatter(X, Y, color='red', s=100, label='Active Poles', zorder=5)
        X_plot = np.linspace(min(X)*0.9, max(X)*1.1, 100)
        ax2.plot(X_plot, m_spec * X_plot + b_spec, 'b--', label='Spectral Fit')
        ax2.set_xlabel(r'$\log(t_j)$')
        ax2.set_ylabel(r'$\log(a_j) + (x_c - x_0)t_j$')
        ax2.set_title(r"Spectral Density for $\gamma$")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Summary text
        ax3.axis('off')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = (rf'Hybrid Method Results:' '\n\n'
                   rf'd-log Padé extracted $x_c$:' '\n'
                   rf'  $x_c$ = {xc_inferred:.4f} (True: {xc_true})' '\n\n'
                   rf'Spectral Density extracted $\gamma$:' '\n'
                   rf'  $\gamma$ = {gamma_inferred:.4f} (True: {gamma_true})')
        ax3.text(0.1, 0.8, textstr, transform=ax3.transAxes, fontsize=14,
                 verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig("vp_hybrid.png")
        plt.close()
    else:
        print("\nNot enough active poles (>1e-8) to perform Spectral Density regression.")


if __name__ == "__main__":
    test_hybrid_pade_spectral()