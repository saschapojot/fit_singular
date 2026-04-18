import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre
from scipy.optimize import lsq_linear


def true_function(x, xc, gamma, c):
    """The theoretical function to fit: f(x) = (xc - x)^(-gamma) + c"""
    return (xc - x) ** (-gamma) + c


def true_derivative(x, xc, gamma):
    """The theoretical derivative: f'(x) = gamma * (xc - x)^(-(gamma + 1))"""
    return gamma * (xc - x) ** (-(gamma + 1))


def test_laguerre_fit():
    # --- 1. Define true parameters and generate synthetic data ---
    xc_true = 5.0  # Critical point (must be > x)
    gamma_true = 1.5  # Critical exponent
    c_true = 2.0  # Constant offset

    x0 = 4.6  # Starting x
    dx = 0.01  # Step size (Delta x)
    M = 20  # Number of data points
    noise_std = 0.01  # Standard deviation of the Gaussian noise

    # Generate x_m and shifted x'_m
    m_indices = np.arange(M)
    x_m = x0 + m_indices * dx
    x_prime_m = m_indices * dx  # Shifted coordinates to prevent overflow

    # Generate true noise-free data and noisy data F
    F_true = true_function(x_m, xc_true, gamma_true, c_true)
    F_prime_true = true_derivative(x_m, xc_true, gamma_true)

    np.random.seed(42)
    noise = np.random.normal(0, noise_std, size=M)
    F = F_true + noise

    # --- 2. Set up the Laguerre Quadrature ---
    n = 10  # Degree of the Laguerre polynomial (number of roots)
    # Get the roots (t_j) and weights of the Laguerre polynomial L_n(t)
    t_j, _ = roots_laguerre(n)

    # --- 3. Construct the Matrix G ---
    # G is an M x (n+1) matrix
    G = np.zeros((M, n + 1))

    # First column is all 1s (for the constant c)
    G[:, 0] = 1.0

    # Remaining columns are exp(t_j[j] * m * dx)
    for j in range(n):
        G[:, j + 1] = np.exp(t_j[j] * x_prime_m)

    # --- 4. Solve the linear system G * A = F with Non-Negative Constraints ---
    # Due to the Laplace transform property of power laws, the coefficients a_j MUST be >= 0.
    # Unconstrained least squares causes massive alternating coefficients (ill-conditioning).
    # We constrain c to be in (-inf, inf) and a_j to be in [0, inf).
    lower_bounds = [-np.inf] + [0.0] * n
    upper_bounds = [np.inf] + [np.inf] * n

    res = lsq_linear(G, F, bounds=(lower_bounds, upper_bounds))
    A = res.x

    c_fit = A[0]
    a_fit = A[1:]

    # Transform coefficients a_j to w_tilde_j for the true coordinates x_m
    # w_tilde_j = a_j * exp(-t_j * x0)
    w_tilde = a_fit * np.exp(-t_j * x0)

    # --- 5. Reconstruct the fitted function and its derivative ---
    F_fit = G @ A

    # Compute the derivative of the fitted function using w_tilde and true coordinates
    F_prime_fit = np.zeros(M)
    for j in range(n):
        F_prime_fit += w_tilde[j] * t_j[j] * np.exp(t_j[j] * x_m)

    # --- 6. Infer xc and gamma ---
    # (f - c) / f' ~ (xc - x) / gamma = (-1/gamma)*x + (xc/gamma)
    ratio = (F_fit - c_fit) / F_prime_fit

    # Fit a line y = mx + b to the ratio
    m_line, b_line = np.polyfit(x_m, ratio, 1)

    # Extract gamma and xc from the slope and intercept
    gamma_inferred = -1.0 / m_line
    xc_inferred = b_line * gamma_inferred

    # --- 7. Calculate Goodness of Fit & Errors ---
    # 7a. Error vs Noisy Data (Function)
    rmse_noisy = np.sqrt(np.mean((F - F_fit) ** 2))
    ss_res = np.sum((F - F_fit) ** 2)
    ss_tot = np.sum((F - np.mean(F)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # 7b. Error vs True (Noise-Free) Function
    rmse_true = np.sqrt(np.mean((F_true - F_fit) ** 2))
    max_err_true = np.max(np.abs(F_true - F_fit))

    # 7c. Error vs True Derivative
    rmse_deriv = np.sqrt(np.mean((F_prime_true - F_prime_fit) ** 2))
    max_err_deriv = np.max(np.abs(F_prime_true - F_prime_fit))

    print(f"Laguerre roots (t_j):\n{t_j}")
    print("-" * 40)
    print(f"True constant c: {c_true:.4f}")
    print(f"Fitted constant c: {c_fit:.4f}")
    print("-" * 40)
    print(f"Fitted coefficients a_j (shifted coords):\n{a_fit}")
    print("-" * 40)
    print(f"Transformed coefficients w_tilde_j (true coords):\n{w_tilde}")
    print("-" * 40)
    print("Function Fit (vs Noisy Data):")
    print(f"RMSE:      {rmse_noisy:.6f}")
    print(f"R-squared: {r_squared:.6f}")
    print("-" * 40)
    print("Function Error (vs True Noise-Free Function):")
    print(f"RMSE:      {rmse_true:.6f}")
    print(f"Max Error: {max_err_true:.6f}")
    print("-" * 40)
    print("Derivative Error (vs True Exact Derivative):")
    print(f"RMSE:      {rmse_deriv:.6f}")
    print(f"Max Error: {max_err_deriv:.6f}")
    print("-" * 40)
    print("Inferred Critical Parameters:")
    print(f"True gamma: {gamma_true:.4f}  |  Inferred gamma: {gamma_inferred:.4f}")
    print(f"True xc:    {xc_true:.4f}  |  Inferred xc:    {xc_inferred:.4f}")
    print("-" * 40)

    # --- 8. Plotting the results ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Function f(x)
    ax1.plot(x_m, F, 'ko', label='Noisy Data', markersize=5)
    ax1.plot(x_m, F_true, 'b-', label='True Function', alpha=0.6)
    ax1.plot(x_m, F_fit, 'r--', label=f'Laguerre Fit (n={n})', linewidth=2)
    ax1.set_title('Function Fit: $f(x)$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')

    textstr1 = (f'vs True Function:\n'
                f'  RMSE = {rmse_true:.5f}\n'
                f'  Max Err = {max_err_true:.5f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Derivative f'(x)
    ax2.plot(x_m, F_prime_true, 'b-', label="True Derivative", alpha=0.6)
    ax2.plot(x_m, F_prime_fit, 'g--', label="Fitted Derivative", linewidth=2)
    ax2.set_title("Derivative Fit: $f'(x)$")
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")

    textstr2 = (f'vs True Derivative:\n'
                f'  RMSE = {rmse_deriv:.5f}\n'
                f'  Max Err = {max_err_deriv:.5f}')
    ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Ratio (f - c) / f'
    ax3.plot(x_m, ratio, 'mo', label='Fitted Ratio Data', markersize=5)
    ax3.plot(x_m, m_line * x_m + b_line, 'k-', label='Linear Regression', linewidth=2)
    ax3.set_title(r"Inference: $\frac{f(x)-c}{f'(x)} \simeq \frac{x_c - x}{\gamma}$")
    ax3.set_xlabel('x')
    ax3.set_ylabel("Ratio")

    # Fixed the escape sequence warning by using an 'r' before the f-string
    textstr3 = (rf'Inferred:' '\n'
                rf'  $\gamma$ = {gamma_inferred:.4f} (True: {gamma_true})' '\n'
                rf'  $x_c$ = {xc_inferred:.4f} (True: {xc_true})')
    ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    ax3.legend(loc='lower left')
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("func_fit.png")
    plt.close()


# Execute the test function
test_laguerre_fit()