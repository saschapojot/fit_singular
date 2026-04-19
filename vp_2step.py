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


def test_varpro_derivative_ratio_fit():
    # --- 1. Define true parameters and generate synthetic data ---
    xc_true = 5.0
    gamma_true = 2.1
    c_true = 2.0

    x0 = 4.2
    x_end = 4.8
    M = 50
    x_m = np.linspace(x0, x_end, M)
    dx = x_m[1] - x_m[0]

    noise_std = 0.1

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
    n = 5  # We can use fewer terms now because they are optimized!
    t_initial, _ = roots_laguerre(n)
    m_diff_indices = np.arange(M - 1)

    def varpro_residual(t_params):
        """
        Given nonlinear parameters t_params, project out the linear parameters a,
        and return the whitened residual.
        """
        # Construct Design Matrix for current t_params
        G_diff = np.zeros((M - 1, n))
        for j in range(n):
            G_diff[:, j] = np.exp(t_params[j] * m_diff_indices * dx) * (np.exp(t_params[j] * dx) - 1.0)

        # Whiten Design Matrix
        G_tilde = solve_triangular(L, G_diff, lower=True)

        # Project out linear parameters 'a' using Non-Negative Least Squares
        res_linear = lsq_linear(G_tilde, d_tilde, bounds=(0.0, np.inf))
        a_opt = res_linear.x

        # Compute and return the residual vector
        residual = d_tilde - G_tilde @ a_opt
        return residual

    # Optimize the nonlinear exponents (t_j)
    res_nonlin = least_squares(varpro_residual, t_initial, bounds=(0.0, 200.0), method='trf')
    t_opt = res_nonlin.x

    # --- 4. Extract Final Linear Parameters ---
    G_diff_opt = np.zeros((M - 1, n))
    for j in range(n):
        G_diff_opt[:, j] = np.exp(t_opt[j] * m_diff_indices * dx) * (np.exp(t_opt[j] * dx) - 1.0)

    G_tilde_opt = solve_triangular(L, G_diff_opt, lower=True)
    a_opt = lsq_linear(G_tilde_opt, d_tilde, bounds=(0.0, np.inf)).x

    # --- 5. Compute Analytical Derivatives f' and f'' ---
    w_tilde = a_opt * np.exp(-t_opt * x0)

    F_prime_fit = np.zeros(M)
    F_double_prime_fit = np.zeros(M)

    for j in range(n):
        F_prime_fit += w_tilde[j] * t_opt[j] * np.exp(t_opt[j] * x_m)
        F_double_prime_fit += w_tilde[j] * (t_opt[j] ** 2) * np.exp(t_opt[j] * x_m)

    # --- 6. Infer xc and gamma (TWO-STEP METHOD) ---
    num_points_fit = 15
    x_m_reg = x_m[-num_points_fit:]

    # Step 6a: Extract xc using the ratio f'/f''
    ratio_fit = F_prime_fit / F_double_prime_fit
    ratio_reg = ratio_fit[-num_points_fit:]
    m_line, b_line = np.polyfit(x_m_reg, ratio_reg, 1)
    xc_inferred = -b_line / m_line

    # Step 6b: Extract gamma using Log-Log regression on f'
    # ln(f'(x)) = -(gamma + 1) * ln(xc - x) + const
    log_dx = np.log(xc_inferred - x_m_reg)
    log_f_prime = np.log(F_prime_fit[-num_points_fit:])

    m_log, b_log = np.polyfit(log_dx, log_f_prime, 1)
    gamma_inferred = -m_log - 1.0

    # --- 7. Print Results ---
    print("-" * 50)
    print("Inferred Critical Parameters (Two-Step VarPro Method):")
    print(f"True xc:    {xc_true:.4f}  |  Inferred xc:    {xc_inferred:.4f}")
    print(f"True gamma: {gamma_true:.4f}  |  Inferred gamma: {gamma_inferred:.4f}")
    print("-" * 50)

    # --- 8. Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: First Derivative
    axs[0, 0].plot(x_m, F_prime_true, 'b-', label="True f'(x)", alpha=0.6)
    axs[0, 0].plot(x_m, F_prime_fit, 'r--', label="Fitted f'(x)", linewidth=2)
    axs[0, 0].set_title("First Derivative: f'(x)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: Second Derivative
    axs[0, 1].plot(x_m, F_double_prime_true, 'b-', label="True f''(x)", alpha=0.6)
    axs[0, 1].plot(x_m, F_double_prime_fit, 'g--', label="Fitted f''(x)", linewidth=2)
    axs[0, 1].set_title("Second Derivative: f''(x)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Ratio for xc
    axs[1, 0].plot(x_m, ratio_fit, 'mo', label='All Ratio Data', markersize=5, alpha=0.4)
    axs[1, 0].plot(x_m_reg, ratio_reg, 'co', label=f'Regression Pts (Last {num_points_fit})', markersize=7)
    axs[1, 0].plot(x_m, m_line * x_m + b_line, 'k-', label='Linear Regression', linewidth=2)
    axs[1, 0].set_title("Ratio: f'(x)/f''(x) (Extracts xc)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: Log-Log for gamma
    axs[1, 1].plot(log_dx, log_f_prime, 'co', label='Log-Log Data', markersize=7)
    axs[1, 1].plot(log_dx, m_log * log_dx + b_log, 'k-', label='Linear Regression', linewidth=2)
    axs[1, 1].set_title("Log-Log Fit: ln(f') vs ln(xc - x) (Extracts gamma)")
    axs[1, 1].set_xlabel("ln(xc - x)")
    axs[1, 1].set_ylabel("ln(f'(x))")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = (f'Inferred:\n'
               f'xc = {xc_inferred:.4f} (True: {xc_true})\n'
               f'gamma = {gamma_inferred:.4f} (True: {gamma_true})')
    axs[1, 1].text(0.05, 0.95, textstr, transform=axs[1, 1].transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("vp_twostep.png")
    plt.close()



test_varpro_derivative_ratio_fit()