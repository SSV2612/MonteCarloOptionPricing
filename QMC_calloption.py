import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import qmc
from scipy.special import erfinv

# Calculation of exact option price for plain vanilla call option
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

# Quasi Monte carlo simulation for plain vanilla call option
def quasi_monte_carlo_simulation_plaincalloption(S, K, T, r, sigma, num_paths, num_steps):
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
    average_option_price: Average option price using Antithetic variance reduction for Monte Carlo simulation
    final_option_prices: Array of estimated call option prices for a pair of paths (2* number of paths) - Antithetic
    """
   dt = T/num_steps

   option_prices = np.zeros(num_paths)
   average_option_price = np.zeros(num_paths)

   for i in range(num_paths):
        S_Path = np.zeros(num_steps+1)
        S_Path[0] = S

        for j in range(1,num_steps+1):
            # generating random numbers using Halton sequence
            sampler = qmc.Halton(d=1,scramble=True)
            halton_paths = sampler.random(1)
            norm_halton_paths = np.sqrt(2)*erfinv((2 * halton_paths) - 1)
            S_Path[j] = S_Path[j - 1] * np.exp((r-0.5*sigma**2) * dt + sigma * np.sqrt(dt) * norm_halton_paths)

        option_prices[i] = max(0, S_Path[-1] - K) * np.exp(-r * T)
  
   average_option_price = np.mean(option_prices)

   return average_option_price, option_prices

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for the given data.
    """
    mean_value = np.mean(data)
    margin_of_error = norm.ppf(1 - (1 - confidence) / 2) * np.std(data) / np.sqrt(len(data))
    lower_bound = mean_value - margin_of_error
    upper_bound = mean_value + margin_of_error
    return lower_bound, upper_bound


S = 100  # stock price S_{0}
K = 110  # strike
T = 2  # time to maturity
r = 0.05  # risk-free rate in annual %
sigma = 0.25  # annual volatility in %
num_steps = 100  # time steps
num_paths = 10**5  # initial number of trials

# Vary the number of paths and store the results
num_paths_list = [400, 1000, 1500, 2000, 3000, 4500, 6000, 9000, 10000]

# can modify the num_paths in below list and check for various number of paths
#num_paths_list = [10000, 25000, 50000, 75000, 10**5]

option_prices_list = []
confidence_intervals = []
absolute_error_list = []

monte, monte_prices = quasi_monte_carlo_simulation_plaincalloption(S, K, T, r, sigma, num_paths, num_steps)
print(f"SDE2 Monte Carlo Option Price is {monte}")
bs = black_scholes_call(S, K, T, r, sigma)
print(f"SDE2 Black Scholes Option Price is {bs}")

# # Calculate errors
absolute_error = abs(monte - bs)
relative_error = abs(monte - bs) / bs
print("Absolute Error:", absolute_error)
print("Relative Error:", relative_error)

for num_paths in num_paths_list:
    # time the code for QMC implementation
    start_time = time.time()
    # plain vanilla call option price calculation using QMC
    average_monte_price, monte_prices = quasi_monte_carlo_simulation_plaincalloption(S, K, T, r, sigma, num_paths, num_steps)
    end_time = time.time()

    option_prices_list.append(average_monte_price)
    absolute_error = abs(average_monte_price - bs)
    absolute_error_list.append(absolute_error)

    # Calculate and store 95% confidence intervals
    lower_bound, upper_bound = calculate_confidence_interval(monte_prices)
    confidence_intervals.append((lower_bound, upper_bound))
    confidence_interval_width = upper_bound - lower_bound
    print(f"95% Confidence Interval for {num_paths} paths: ({lower_bound}, {upper_bound})")
    print(f"95% C.I width for {num_paths}:{confidence_interval_width}")


# Plot the results with vertical lines for confidence intervals
option_prices_list = np.array(option_prices_list)
confidence_intervals = np.array(confidence_intervals).T
absolute_error_list = np.array(absolute_error_list)

plt.figure(figsize=(10, 6))
plt.scatter(num_paths_list, option_prices_list, label="Quasi Monte Carlo Option Price", marker='o')
plt.axhline(y=black_scholes_call(S, K, T, r, sigma), color='r', linestyle='--', label="Black Scholes Option Price")

# Draw vertical lines for confidence intervals
for i, num_paths in enumerate(num_paths_list):
    plt.vlines(num_paths, confidence_intervals[0, i], confidence_intervals[1, i], colors='blue', linestyles='dashed', alpha=0.7, label=f"{0.95*100}% CI")

plt.title("Quasi Monte Carlo Call Option Price vs. Number of Paths with 95% Confidence Intervals")
plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.show()