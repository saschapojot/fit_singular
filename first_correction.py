import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre
from scipy.optimize import least_squares, lsq_linear, curve_fit
from scipy.linalg import cholesky, solve_triangular


def true_function(x, xc, gamma, delta, c):
    """f(x) = (xc - x)^(-gamma) + (xc - x)^(-gamma + delta) + c"""
    return (xc - x) ** (-gamma) + (xc - x) ** (-gamma + delta) + c


def test_hybrid_pade_nlls_correction():
    # --- 1. Define true parameters and generate synthetic data ---
    xc_true = 5.0
    gamma_true = 2.1
    delta_true = 0.8
    c_true = 2.0

    x0 = 4.0
    x_end = 4.8
    M = 80
    x_m = np.linspace(x0, x_end, M)
    dx = x_m[1] - x_m[0]

    noise_std = 0.01# Lower noise to allow delta extraction
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

    # --- 3. VARIABLE PROJECTION (Golub-Pereyra) ---
    n = 6
    t_initial, _ = roots_laguerre(n)
    m_diff_indices = np.arange(M - 1)
    lambda_reg = 1e-6

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

    # --- 4. Compute Analytical Derivatives f' and f'' ---
    F_prime_fit = np.zeros(M)
    F_double_prime_fit = np.zeros(M)
    for j in range(n):
        F_prime_fit += w_tilde[j] * t_opt[j] * np.exp(t_opt[j] * x_m)
        F_double_prime_fit += w_tilde[j] * (t_opt[j] ** 2) * np.exp(t_opt[j] * x_m)

    # --- 5. Infer initial xc and gamma using d-log Padé ---
    ratio_fit = F_prime_fit / F_double_prime_fit

    # Use points closest to xc for the leading term
    num_pts_lead = 10
    x_lead = x_m[-num_pts_lead:]
    ratio_lead = ratio_fit[-num_pts_lead:]

    m_line, b_line = np.polyfit(x_lead, ratio_lead, 1)
    gamma_pade = (-1.0 / m_line) - 1.0
    xc_pade = -b_line / m_line

    # --- 6. Infer initial delta using the deviation Delta R ---
    # Calculate deviation for all points
    delta_R = ratio_fit - (xc_pade - x_m) / (gamma_pade + 1.0)

    # Select points close to xc for the correction term as well
    num_pts_corr = 20
    x_corr = x_m[-num_pts_corr:]
    dR_corr = np.abs(delta_R[-num_pts_corr:])

    log_dx = np.log(xc_pade - x_corr)
    log_dR = np.log(dR_corr)

    m_log, b_log = np.polyfit(log_dx, log_dR, 1)
    delta_pade = m_log - 1.0

    # Fallback if delta is non-physical (must be > 0)
    if delta_pade <= 0:
        delta_pade = 0.1

    # Estimate c
    c_pade = F_noisy[0] - (xc_pade - x_m[0]) ** (-gamma_pade) - (xc_pade - x_m[0]) ** (-gamma_pade + delta_pade)

    print("-" * 50)
    print("Initial Guesses from d-log Padé & Log-Log Fit:")
    print(f"xc_pade:    {xc_pade:.4f}")
    print(f"gamma_pade: {gamma_pade:.4f}")
    print(f"delta_pade: {delta_pade:.4f}")
    print(f"c_pade:     {c_pade:.4f}")

    # --- 7. Direct 4-Parameter Non-Linear Least Squares (NLLS) ---
    p0 = [xc_pade, gamma_pade, delta_pade, c_pade]
    bounds = ([x_m[-1] + 1e-5, 0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(true_function, x_m, F_noisy, p0=p0, bounds=bounds)
        xc_nlls, gamma_nlls, delta_nlls, c_nlls = popt

        print("-" * 50)
        print("Final Inferred Parameters (Direct 4-Parameter NLLS):")
        print(f"True xc:    {xc_true:.4f}  |  NLLS xc:    {xc_nlls:.4f}")
        print(f"True gamma: {gamma_true:.4f}  |  NLLS gamma: {gamma_nlls:.4f}")
        print(f"True delta: {delta_true:.4f}  |  NLLS delta: {delta_nlls:.4f}")
        print(f"True c:     {c_true:.4f}  |  NLLS c:     {c_nlls:.4f}")
        print("-" * 50)

        F_nlls_fit = true_function(x_m, xc_nlls, gamma_nlls, delta_nlls, c_nlls)
    except Exception as e:
        print(f"NLLS Fit failed: {e}")
        F_nlls_fit = np.zeros_like(x_m)

    # --- 8. Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Leading Term Ratio Fit
    ax1.plot(x_m, ratio_fit, 'mo', label='All Ratio Data', markersize=4, alpha=0.4)
    ax1.plot(x_lead, ratio_lead, 'co', label='Leading Term Reg Pts', markersize=6)
    ax1.plot(x_m, m_line * x_m + b_line, 'k-', label='Linear Fit', linewidth=2)
    ax1.set_title(r"Step 1: Ratio $\frac{f'(x)}{f''(x)}$")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Correction Term Log-Log Fit
    ax2.plot(log_dx, log_dR, 'go', label=r'Deviation $\log|\Delta R|$', markersize=5)
    ax2.plot(log_dx, m_log * log_dx + b_log, 'k-', label=f'Slope = {m_log:.2f}', linewidth=2)
    ax2.set_title(r"Step 2: Log-Log Fit for $\delta$")
    ax2.set_xlabel(r"$\log(x_c - x)$")
    ax2.set_ylabel(r"$\log|\Delta R|$")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Final NLLS Fit
    ax3.plot(x_m, F_noisy, 'ko', label='Noisy Data', markersize=4, alpha=0.6)
    ax3.plot(x_m, F_true, 'b-', label='True Function', linewidth=2, alpha=0.7)
    if np.any(F_nlls_fit):
        ax3.plot(x_m, F_nlls_fit, 'r--', label='4-Param NLLS Fit', linewidth=2)
    ax3.set_title("Step 3: Final NLLS Fit")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig("1st_correction.png")
    plt.close()

if __name__ == "__main__":
    test_hybrid_pade_nlls_correction()