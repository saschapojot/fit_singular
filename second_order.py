import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre
from scipy.optimize import lsq_linear
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


def test_derivative_ratio_fit():
    # --- 1. Define true parameters and generate synthetic data ---
    xc_true = 5.0
    gamma_true = 1.5
    c_true = 2.0

    x0 = 4.6
    x_end = 4.8
    M = 100
    x_m = np.linspace(x0, x_end, M)
    dx = x_m[1] - x_m[0]
    noise_std = 0.1

    m_indices = np.arange(M)
    x_m = x0 + m_indices * dx

    # Generate noisy data
    np.random.seed(42)
    F_true = true_function(x_m, xc_true, gamma_true, c_true)
    F_noisy = F_true + np.random.normal(0, noise_std, size=M)

    # True derivatives for comparison
    F_prime_true = true_first_derivative(x_m, xc_true, gamma_true)
    F_double_prime_true = true_second_derivative(x_m, xc_true, gamma_true)

    # --- 2. Set up the Laguerre Quadrature ---
    n = 10
    t_j, _ = roots_laguerre(n)

    # --- 3. Construct the Differenced Data and Design Matrix ---
    # d_m = f_{m+1} - f_m
    d = F_noisy[1:] - F_noisy[:-1]

    # G_diff[m, j] = exp(t_j * m * dx) * (exp(t_j * dx) - 1)
    G_diff = np.zeros((M - 1, n))
    for j in range(n):
        m_diff_indices = np.arange(M - 1)
        G_diff[:, j] = np.exp(t_j[j] * m_diff_indices * dx) * (np.exp(t_j[j] * dx) - 1.0)

    # --- 4. Whiten the Data (Generalized Least Squares) ---
    # Covariance matrix V for differenced i.i.d noise (MA(1) process)
    V = np.diag(np.full(M - 1, 2.0)) + \
        np.diag(np.full(M - 2, -1.0), k=1) + \
        np.diag(np.full(M - 2, -1.0), k=-1)

    # Cholesky decomposition: V = L * L^T
    L = cholesky(V, lower=True)

    # Whiten the data and design matrix: L * x = b => x = L^{-1} * b
    d_tilde = solve_triangular(L, d, lower=True)
    G_tilde = solve_triangular(L, G_diff, lower=True)

    # --- 5. Solve the linear system with Non-Negative Constraints ---
    lower_bounds = [0.0] * n
    upper_bounds = [np.inf] * n
    res = lsq_linear(G_tilde, d_tilde, bounds=(lower_bounds, upper_bounds))
    a_fit = res.x

    # --- 6. Compute Analytical Derivatives f' and f'' ---
    # Transform coefficients: w_j = a_j * exp(-t_j * x0)
    w_tilde = a_fit * np.exp(-t_j * x0)

    F_prime_fit = np.zeros(M)
    F_double_prime_fit = np.zeros(M)

    for j in range(n):
        # f'(x) = sum w_j * t_j * exp(t_j * x)
        F_prime_fit += w_tilde[j] * t_j[j] * np.exp(t_j[j] * x_m)
        # f''(x) = sum w_j * t_j^2 * exp(t_j * x)
        F_double_prime_fit += w_tilde[j] * (t_j[j] ** 2) * np.exp(t_j[j] * x_m)

    # --- 7. Infer xc and gamma using the ratio f'/f'' ---
    # Ratio: f'/f'' = (xc - x) / (gamma + 1) = (-1 / (gamma + 1)) * x + (xc / (gamma + 1))
    ratio_fit = F_prime_fit / F_double_prime_fit

    # Use the last few points where the singularity dominates the behavior
    num_points_fit = 6
    x_m_reg = x_m[-num_points_fit:]
    ratio_reg = ratio_fit[-num_points_fit:]

    # Linear regression: y = mx + b
    m_line, b_line = np.polyfit(x_m_reg, ratio_reg, 1)

    # Extract parameters
    # m = -1 / (gamma + 1)  =>  gamma = -1/m - 1
    gamma_inferred = (-1.0 / m_line) - 1.0

    # b = xc / (gamma + 1)  =>  xc = b * (gamma + 1) = -b / m
    xc_inferred = -b_line / m_line

    # --- 8. Print Results ---
    print("-" * 40)
    print("Inferred Critical Parameters (from f'/f'' ratio):")
    print(f"True gamma: {gamma_true:.4f}  |  Inferred gamma: {gamma_inferred:.4f}")
    print(f"True xc:    {xc_true:.4f}  |  Inferred xc:    {xc_inferred:.4f}")
    print("-" * 40)

    # --- 9. Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: First Derivative
    ax1.plot(x_m, F_prime_true, 'b-', label="True $f'(x)$", alpha=0.6)
    ax1.plot(x_m, F_prime_fit, 'r--', label="Fitted $f'(x)$", linewidth=2)
    ax1.set_title("First Derivative: $f'(x)$")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Second Derivative
    ax2.plot(x_m, F_double_prime_true, 'b-', label="True $f''(x)$", alpha=0.6)
    ax2.plot(x_m, F_double_prime_fit, 'g--', label="Fitted $f''(x)$", linewidth=2)
    ax2.set_title("Second Derivative: $f''(x)$")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Ratio and Regression
    ax3.plot(x_m, ratio_fit, 'mo', label='All Ratio Data', markersize=5, alpha=0.4)
    ax3.plot(x_m_reg, ratio_reg, 'co', label=f'Regression Pts (Last {num_points_fit})', markersize=7)

    # Plot the regression line extended across the domain
    ax3.plot(x_m, m_line * x_m + b_line, 'k-', label='Linear Regression', linewidth=2)
    ax3.set_title(r"Ratio: $\frac{f'(x)}{f''(x)} = \frac{x_c - x}{\gamma + 1}$")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = (rf'Inferred:' '\n'
               rf'  $\gamma$ = {gamma_inferred:.4f} (True: {gamma_true})' '\n'
               rf'  $x_c$ = {xc_inferred:.4f} (True: {xc_true})')
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("2nd.png")
    plt.close()



test_derivative_ratio_fit()