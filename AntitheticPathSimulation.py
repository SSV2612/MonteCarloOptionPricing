import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from mrg32k3a.mrg32k3a import MRG32k3a

# Implmenting the Selected MRNG
rng = MRG32k3a(ref_seed=(12345, 54321, 1388, 54321, 12345, 99211))
x = rng.normalvariate(mu=0, sigma=1)
 

def black_scholes_call(S, K, T, r, sigma):
    """
    Inputs
    S: Current stock Price
    K: Strike Price
    T: Time to maturity (1 year = 1, 1 month = 1/12)
    r: Risk-free interest rate
    sigma: Volatility
    
    Output
    call_price: Value of the option
    """
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        raise ValueError("Invalid input values. Ensure positive values for T, S, K, and sigma.")
 
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
   
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price
 
def monte_carlo_simulation_SDE2(S, K, T, r, sigma, num_paths, num_steps):
    """
    Inputs:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility
    num_paths: Number of Monte Carlo paths to simulate
    num_steps: Number of time steps
   
    Output:
    call_prices: Array of normal call option prices for each path
    """
    dt = T / num_steps
 
    call_prices = np.zeros(num_paths)
 
    for i in range(num_paths):
        X_path = np.zeros(num_steps + 1)
        X_path_Anti = np.zeros(num_steps + 1)
        X_path[0] = S
        X_path_Anti[0] = S
 
        for j in range(1, num_steps + 1):
            # standard_normal_array = np.random.normal(0, 1, 1)
            standard_normal_array = rng.normalvariate(mu=0, sigma=1)
            anti_standard_normal_array = - standard_normal_array
            X_path[j] = X_path[j - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * standard_normal_array)
            X_path_Anti[j] = X_path_Anti[j - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * anti_standard_normal_array)
 
    return X_path, X_path_Anti
 

# Example usage
S = 100  # Stock price S_{0}
K = 110  # Strike
T = 2  # Time to maturity
r = 0.05  # Risk-free rate in annual %
sigma = 0.25  # Annual volatility in %
num_steps = 100  # time steps
num_paths = 1  # number of trials
 

Good_path, Bad_Path = monte_carlo_simulation_SDE2(S, K, T, r, sigma, num_paths, num_steps)
 

plt.figure(figsize=(10, 6))
plt.plot(Good_path , label='Standard Path')
plt.plot(Bad_Path , label='Anti Path')
plt.title("Antithetic path for stock price")
plt.xlabel("Iteration")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
 