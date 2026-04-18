import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre
from scipy.optimize import lsq_linear
from scipy.linalg import cholesky, solve_triangular


def true_function(x, xc, gamma, c):
    """The theoretical function to fit: f(x) = (xc - x)^(-gamma) + c"""
    return (xc - x) ** (-gamma) + c


def true_derivative(x, xc, gamma):
    """The theoretical derivative: f'(x) = gamma * (xc - x)^(-(gamma + 1))"""
    return gamma * (xc - x) ** (-(gamma + 1))


def test_laguerre_fit():
    # --- 1. Define true parameters and generate synthetic data ---
    xc_true = 5.0
    gamma_true = 1.5
    c_true = 2.0

    # x0 = 4.6
    # dx = 0.01
    # M = 20
    x0=4.2
    x_end=4.8
    M=100
    x_m=np.linspace(x0,x_end,M)
    dx=x_m[1]-x_m[0]
    noise_std = 0.01

    m_indices = np.arange(M)

    x_prime_m = m_indices * dx

    F_true = true_function(x_m, xc_true, gamma_true, c_true)
    F_prime_true = true_derivative(x_m, xc_true, gamma_true)

    np.random.seed(42)
    noise = np.random.normal(0, noise_std, size=M)
    F = F_true + noise

    # --- 2. Set up the Laguerre Quadrature ---
    n = 10
    t_j, _ = roots_laguerre(n)

    # --- 3. Construct the Differenced Data and Design Matrix ---
    # Calculate f_{m+1} - f_m
    d = F[1:] - F[:-1]

    # Construct G_diff for the differenced equation
    # G_diff[m, j] = exp(t_j * m * dx) * (exp(t_j * dx) - 1)
    G_diff = np.zeros((M - 1, n))
    for j in range(n):
        # Note: m goes from 0 to M-2 for the differences
        m_diff_indices = np.arange(M - 1)
        G_diff[:, j] = np.exp(t_j[j] * m_diff_indices * dx) * (np.exp(t_j[j] * dx) - 1.0)

    # --- 4. Whiten the Data (Generalized Least Squares) ---
    # Construct the covariance matrix V = D * D^T
    # It has 2 on the diagonal, and -1 on the first off-diagonals
    V = np.diag(np.full(M - 1, 2.0)) + \
        np.diag(np.full(M - 2, -1.0), k=1) + \
        np.diag(np.full(M - 2, -1.0), k=-1)

    # Cholesky decomposition: V = L * L^T
    L = cholesky(V, lower=True)

    # Whiten the data and design matrix by solving L * x = b  =>  x = L^{-1} * b
    d_tilde = solve_triangular(L, d, lower=True)
    G_tilde = solve_triangular(L, G_diff, lower=True)

    # --- 5. Solve the linear system with Non-Negative Constraints ---
    lower_bounds = [0.0] * n
    upper_bounds = [np.inf] * n

    res = lsq_linear(G_tilde, d_tilde, bounds=(lower_bounds, upper_bounds))
    a_fit = res.x

    # --- 6. Recover the constant c and reconstruct ---
    # Now that we have a_fit, we can calculate the exponential part of the original function
    G_orig = np.zeros((M, n))
    for j in range(n):
        G_orig[:, j] = np.exp(t_j[j] * x_prime_m)

    exp_part = G_orig @ a_fit

    # c is simply the mean of the residuals (F - exp_part)
    c_fit = np.mean(F - exp_part)

    # Transform coefficients for derivative calculation
    w_tilde = a_fit * np.exp(-t_j * x0)

    F_fit = exp_part + c_fit

    F_prime_fit = np.zeros(M)
    for j in range(n):
        F_prime_fit += w_tilde[j] * t_j[j] * np.exp(t_j[j] * x_m)

    # --- 7. Infer xc and gamma ---
    ratio = (F_fit - c_fit) / F_prime_fit

    # Use the last few points for better asymptotic accuracy
    num_points_fit = 5
    x_m_fit = x_m[-num_points_fit:]
    ratio_fit = ratio[-num_points_fit:]

    m_line, b_line = np.polyfit(x_m_fit, ratio_fit, 1)

    gamma_inferred = -1.0 / m_line
    xc_inferred = b_line * gamma_inferred

    # --- 8. Calculate Goodness of Fit & Errors ---
    rmse_noisy = np.sqrt(np.mean((F - F_fit) ** 2))
    rmse_true = np.sqrt(np.mean((F_true - F_fit) ** 2))
    rmse_deriv = np.sqrt(np.mean((F_prime_true - F_prime_fit) ** 2))

    print(f"True constant c: {c_true:.4f}")
    print(f"Recovered c:     {c_fit:.4f}")
    print("-" * 40)
    print("Function Fit (vs Noisy Data):")
    print(f"RMSE:      {rmse_noisy:.6f}")
    print("-" * 40)
    print("Derivative Error (vs True Exact Derivative):")
    print(f"RMSE:      {rmse_deriv:.6f}")
    print("-" * 40)
    print("Inferred Critical Parameters:")
    print(f"True gamma: {gamma_true:.4f}  |  Inferred gamma: {gamma_inferred:.4f}")
    print(f"True xc:    {xc_true:.4f}  |  Inferred xc:    {xc_inferred:.4f}")
    print("-" * 40)

    # --- 9. Plotting the results ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(x_m, F, 'ko', label='Noisy Data', markersize=5)
    ax1.plot(x_m, F_true, 'b-', label='True Function', alpha=0.6)
    ax1.plot(x_m, F_fit, 'r--', label=f'Whitened Diff Fit', linewidth=2)
    ax1.set_title('Function Fit: $f(x)$')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x_m, F_prime_true, 'b-', label="True Derivative", alpha=0.6)
    ax2.plot(x_m, F_prime_fit, 'g--', label="Fitted Derivative", linewidth=2)
    ax2.set_title("Derivative Fit: $f'(x)$")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(x_m, ratio, 'mo', label='All Ratio Data', markersize=5, alpha=0.4)
    ax3.plot(x_m_fit, ratio_fit, 'co', label=f'Last {num_points_fit} Pts', markersize=7)
    ax3.plot(x_m, m_line * x_m + b_line, 'k-', label='Linear Regression', linewidth=2)
    ax3.set_title(r"Inference: $\frac{f(x)-c}{f'(x)} \simeq \frac{x_c - x}{\gamma}$")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr3 = (rf'Inferred:' '\n'
                rf'  $\gamma$ = {gamma_inferred:.4f} (True: {gamma_true})' '\n'
                rf'  $x_c$ = {xc_inferred:.4f} (True: {xc_true})')
    ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("func_fit.png")
    plt.close()


test_laguerre_fit()