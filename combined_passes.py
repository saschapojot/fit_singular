import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre, roots_genlaguerre
from scipy.optimize import least_squares, lsq_linear, curve_fit
from scipy.linalg import cholesky, solve_triangular


def true_function(x, xc, gamma, delta, c):
    """f(x) = (xc - x)^(-gamma) + (xc - x)^(-gamma + delta) + c"""
    return (xc - x) ** (-gamma) + (xc - x) ** (-gamma + delta) + c


def perform_varpro_and_nlls(x_m, F_noisy, dx, d_tilde, L, t_initial, n, lambda_reg, pass_name="",
                            optimize_scale_only=False):
    """Encapsulates the VARPRO, d-log Padé, and NLLS steps to allow easy repetition."""
    M = len(x_m)
    x0 = x_m[0]
    m_diff_indices = np.arange(M - 1)

    # --- 1. VARIABLE PROJECTION (Golub-Pereyra) ---
    if optimize_scale_only:
        # Optimize only a single global scale factor 'b' to preserve the exact quadrature spacing
        def varpro_residual(b_params):
            t_params = b_params[0] * t_initial
            G_diff = np.zeros((M - 1, n))
            for j in range(n):
                G_diff[:, j] = np.exp(t_params[j] * m_diff_indices * dx) * (np.exp(t_params[j] * dx) - 1.0)
            G_tilde = solve_triangular(L, G_diff, lower=True)
            G_aug = np.vstack((G_tilde, lambda_reg * np.eye(n)))
            d_aug = np.concatenate((d_tilde, np.zeros(n)))
            res_linear = lsq_linear(G_aug, d_aug, bounds=(0.0, np.inf))
            return d_tilde - G_tilde @ res_linear.x

        res_nonlin = least_squares(varpro_residual, [1.0], bounds=(1e-3, 200.0), method='trf')
        t_opt = res_nonlin.x[0] * t_initial
    else:
        # Freely optimize all n nodes independently
        def varpro_residual(t_params):
            G_diff = np.zeros((M - 1, n))
            for j in range(n):
                G_diff[:, j] = np.exp(t_params[j] * m_diff_indices * dx) * (np.exp(t_params[j] * dx) - 1.0)
            G_tilde = solve_triangular(L, G_diff, lower=True)
            G_aug = np.vstack((G_tilde, lambda_reg * np.eye(n)))
            d_aug = np.concatenate((d_tilde, np.zeros(n)))
            res_linear = lsq_linear(G_aug, d_aug, bounds=(0.0, np.inf))
            return d_tilde - G_tilde @ res_linear.x

        res_nonlin = least_squares(varpro_residual, t_initial, bounds=(1e-3, 200.0), method='trf')
        t_opt = res_nonlin.x

    # Extract Linear Parameters
    G_diff_opt = np.zeros((M - 1, n))
    for j in range(n):
        G_diff_opt[:, j] = np.exp(t_opt[j] * m_diff_indices * dx) * (np.exp(t_opt[j] * dx) - 1.0)
    G_tilde_opt = solve_triangular(L, G_diff_opt, lower=True)
    G_aug_opt = np.vstack((G_tilde_opt, lambda_reg * np.eye(n)))
    d_aug_opt = np.concatenate((d_tilde, np.zeros(n)))
    a_opt = lsq_linear(G_aug_opt, d_aug_opt, bounds=(0.0, np.inf)).x
    w_tilde = a_opt * np.exp(-t_opt * x0)

    # --- 2. Compute Analytical Derivatives f' and f'' ---
    F_prime_fit = np.zeros(M)
    F_double_prime_fit = np.zeros(M)
    for j in range(n):
        F_prime_fit += w_tilde[j] * t_opt[j] * np.exp(t_opt[j] * x_m)
        F_double_prime_fit += w_tilde[j] * (t_opt[j] ** 2) * np.exp(t_opt[j] * x_m)

    # --- 3. Infer initial xc and gamma using d-log Padé ---
    ratio_fit = F_prime_fit / F_double_prime_fit

    # Use points close to xc, avoiding the extreme boundary to prevent artifacts
    start_idx_lead = M - 25
    end_idx_lead = M - 5
    x_lead = x_m[start_idx_lead:end_idx_lead]
    ratio_lead = ratio_fit[start_idx_lead:end_idx_lead]

    m_line, b_line = np.polyfit(x_lead, ratio_lead, 1)
    gamma_pade = (-1.0 / m_line) - 1.0
    xc_pade = -b_line / m_line

    # --- 4. Infer initial delta using the deviation Delta R ---
    delta_R = ratio_fit - (xc_pade - x_m) / (gamma_pade + 1.0)

    start_idx_corr = M - 30
    end_idx_corr = M - 10
    x_corr = x_m[start_idx_corr:end_idx_corr]
    dR_corr = np.abs(delta_R[start_idx_corr:end_idx_corr])

    log_dx = np.log(xc_pade - x_corr)
    log_dR = np.log(dR_corr)

    m_log, b_log = np.polyfit(log_dx, log_dR, 1)
    delta_pade = m_log - 1.0

    if delta_pade <= 0:
        delta_pade = 0.1

    c_pade = F_noisy[0] - (xc_pade - x_m[0]) ** (-gamma_pade) - (xc_pade - x_m[0]) ** (-gamma_pade + delta_pade)

    print(f"\n--- {pass_name}: Initial Guesses from d-log Padé & Log-Log Fit ---")
    print(f"xc_pade:    {xc_pade:.4f}")
    print(f"gamma_pade: {gamma_pade:.4f}")
    print(f"delta_pade: {delta_pade:.4f}")
    print(f"c_pade:     {c_pade:.4f}")

    # --- 4.5 Reconstruct the smoothed VARPRO signal ---
    F_varpro_smoothed = np.zeros(M)
    for j in range(n):
        F_varpro_smoothed += w_tilde[j] * np.exp(t_opt[j] * x_m)

    # Align the DC offset with the noisy data at the end point
    F_varpro_smoothed += (F_noisy[-1] - F_varpro_smoothed[-1])

    # --- 5. Direct 4-Parameter NLLS (Fit to Smoothed Data) ---
    xc_pade = max(xc_pade, x_m[-1] + 1e-3)
    gamma_pade = max(gamma_pade, 0.1)

    p0 = [xc_pade, gamma_pade, delta_pade, c_pade]
    bounds = ([x_m[-1] + 1e-5, 0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    try:
        # Fit against the smoothed VARPRO signal instead of raw noisy data
        popt, pcov = curve_fit(true_function, x_m, F_varpro_smoothed, p0=p0, bounds=bounds)
        xc_nlls, gamma_nlls, delta_nlls, c_nlls = popt
        F_nlls_fit = true_function(x_m, xc_nlls, gamma_nlls, delta_nlls, c_nlls)
    except Exception as e:
        print(f"NLLS Fit failed: {e}")
        popt = p0
        F_nlls_fit = np.zeros_like(x_m)

    return popt, F_nlls_fit, ratio_fit, x_lead, ratio_lead, m_line, b_line, log_dx, log_dR, m_log, b_log


def test_hybrid_pade_nlls_correction():
    # --- 1. Define true parameters and generate synthetic data ---
    xc_true = 5.0
    gamma_true = 3.1
    delta_true = 1.5
    c_true = 2.0

    x0 = 4.0
    x_end = 4.8
    M = 80
    x_m = np.linspace(x0, x_end, M)
    dx = x_m[1] - x_m[0]

    noise_std = 0.01
    print(f"noise_std={noise_std}")

    np.random.seed(42)
    F_true = true_function(x_m, xc_true, gamma_true, delta_true, c_true)
    F_noisy = F_true + np.random.normal(0, noise_std, size=M)

    # --- 2. Construct Differenced Data and Whiten ---
    d = F_noisy[1:] - F_noisy[:-1]
    V = np.diag(np.full(M - 1, 2.0)) + \
        np.diag(np.full(M - 2, -1.0), k=1) + \
        np.diag(np.full(M - 2, -1.0), k=-1)
    L = cholesky(V, lower=True)
    d_tilde = solve_triangular(L, d, lower=True)

    n = 7
    lambda_reg = 1e-6

    # ==========================================
    # PASS 1: Standard Laguerre Quadrature
    # ==========================================
    t_initial_1, _ = roots_laguerre(n)

    # In Pass 1, we freely optimize the nodes to get a rough estimate
    popt_1, _, _, _, _, _, _, _, _, _, _ = perform_varpro_and_nlls(
        x_m, F_noisy, dx, d_tilde, L, t_initial_1, n, lambda_reg, pass_name="PASS 1", optimize_scale_only=False
    )
    xc_1, gamma_1, delta_1, c_1 = popt_1

    print("-" * 50)
    print("Pass 1 Final Inferred Parameters:")
    print(f"NLLS xc:    {xc_1:.4f}")
    print(f"NLLS gamma: {gamma_1:.4f}")
    print(f"NLLS delta: {delta_1:.4f}")
    print(f"NLLS c:     {c_1:.4f}")
    print("-" * 50)

    # ==========================================
    # PASS 2: Generalized Laguerre Quadrature
    # ==========================================
    # Use gamma and delta from Pass 1 to set alpha for generalized Laguerre
    n_lead = n // 2
    n_corr = n - n_lead

    # Ensure alpha > -1 (requirement for generalized Laguerre polynomials)
    alpha_lead = max(gamma_1 - 1.0, -0.99)
    alpha_corr = max(gamma_1 - delta_1 - 1.0, -0.99)

    t_lead, _ = roots_genlaguerre(n_lead, alpha_lead)
    t_corr, _ = roots_genlaguerre(n_corr, alpha_corr)
    t_initial_2 = np.concatenate([t_lead, t_corr])

    # In Pass 2, set optimize_scale_only=True to preserve the generalized Laguerre spacing
    popt_2, F_nlls_fit_2, ratio_fit, x_lead, ratio_lead, m_line, b_line, log_dx, log_dR, m_log, b_log = perform_varpro_and_nlls(
        x_m, F_noisy, dx, d_tilde, L, t_initial_2, n, lambda_reg, pass_name="PASS 2", optimize_scale_only=True
    )
    xc_2, gamma_2, delta_2, c_2 = popt_2

    print("-" * 50)
    print("Pass 2 Final Inferred Parameters (Direct 4-Parameter NLLS):")
    print(f"True xc:    {xc_true:.4f}  |  NLLS xc:    {xc_2:.4f}")
    print(f"True gamma: {gamma_true:.4f}  |  NLLS gamma: {gamma_2:.4f}")
    print(f"True delta: {delta_true:.4f}  |  NLLS delta: {delta_2:.4f}")
    print(f"True c:     {c_true:.4f}  |  NLLS c:     {c_2:.4f}")
    print("-" * 50)

    # --- Plotting (Using Pass 2 Results) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Leading Term Ratio Fit
    ax1.plot(x_m, ratio_fit, 'mo', label='All Ratio Data', markersize=4, alpha=0.4)
    ax1.plot(x_lead, ratio_lead, 'co', label='Leading Term Reg Pts', markersize=6)
    ax1.plot(x_m, m_line * x_m + b_line, 'k-', label='Linear Fit', linewidth=2)
    ax1.set_title(r"Pass 2: Ratio $\frac{f'(x)}{f''(x)}$")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Correction Term Log-Log Fit
    ax2.plot(log_dx, log_dR, 'go', label=r'Deviation $\log|\Delta R|$', markersize=5)
    ax2.plot(log_dx, m_log * log_dx + b_log, 'k-', label=f'Slope = {m_log:.2f}', linewidth=2)
    ax2.set_title(r"Pass 2: Log-Log Fit for $\delta$")
    ax2.set_xlabel(r"$\log(x_c - x)$")
    ax2.set_ylabel(r"$\log|\Delta R|$")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Final NLLS Fit
    ax3.plot(x_m, F_noisy, 'ko', label='Noisy Data', markersize=4, alpha=0.6)
    ax3.plot(x_m, F_true, 'b-', label='True Function', linewidth=2, alpha=0.7)
    if np.any(F_nlls_fit_2):
        ax3.plot(x_m, F_nlls_fit_2, 'r--', label='Pass 2 NLLS Fit', linewidth=2)
    ax3.set_title("Pass 2: Final NLLS Fit")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("2passes.png")
    plt.close()


if __name__ == "__main__":
    test_hybrid_pade_nlls_correction()