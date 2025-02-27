import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from mrg32k3a.mrg32k3a import MRG32k3a

# Record the start time
start_time = time.time()

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
        X_path[0] = S

        for j in range(1, num_steps + 1):
            # standard_normal_array = np.random.normal(0, 1, 1)
            standard_normal_array = rng.normalvariate(mu=0, sigma=1)
            X_path[j] = X_path[j - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * standard_normal_array)
            
        # Calculate the option price at maturity (T)
        call_prices[i] = max(0, X_path[-1] - K) * np.exp(-r * T)

    # Calculate the average of all normal call option prices
    average_call_price = np.mean(call_prices)

    return average_call_price, call_prices

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for the given data.
    """
    mean_value = np.mean(data)
    margin_of_error = norm.ppf(1 - (1 - confidence) / 2) * np.std(data) / np.sqrt(len(data))
    lower_bound = mean_value - margin_of_error
    upper_bound = mean_value + margin_of_error
    return lower_bound, upper_bound

# Example usage
S = 100  # Stock price S_{0}
K = 110  # Strike
T = 2  # Time to maturity
r = 0.05  # Risk-free rate in annual %
sigma = 0.25  # Annual volatility in %
num_steps = 100  # time steps
num_paths = 100000  # number of trials

# Vary the number of paths and store the results
# num_paths_list = [400, 1000, 1500, 2000, 3000, 4500, 6000]
num_paths_list = [50000, 100000, 200000, 400000, 640000]
option_prices_list = []
confidence_intervals = []
absolute_error_list = []

monte, monte_prices = monte_carlo_simulation_SDE2(S, K, T, r, sigma, num_paths, num_steps)
print(f"Monte Carlo Option Price is {monte}")
bs = black_scholes_call(S, K, T, r, sigma)
print(f"Black Scholes Option Price is {bs}")

# Calculate errors
absolute_error = abs(monte - bs)
relative_error = abs(monte - bs) / bs
print("Absolute Error:", absolute_error)
print("Relative Error:", relative_error)

for num_paths in num_paths_list:
    average_monte_price, monte_prices = monte_carlo_simulation_SDE2(S, K, T, r, sigma, num_paths, num_steps)
    option_prices_list.append(average_monte_price)
    absolute_error = abs(average_monte_price - bs)
    absolute_error_list.append(absolute_error)

    # Calculate and store 95% confidence intervals
    lower_bound, upper_bound = calculate_confidence_interval(monte_prices)
    confidence_intervals.append((lower_bound, upper_bound))
    print(f"95% Confidence Interval for {num_paths} paths: ({lower_bound}, {upper_bound})")

# Plot the results with vertical lines for confidence intervals
option_prices_list = np.array(option_prices_list)
confidence_intervals = np.array(confidence_intervals).T
absolute_error_list = np.array(absolute_error_list)

plt.figure(figsize=(10, 6))
plt.scatter(num_paths_list, option_prices_list, label="Monte Carlo Option Price", marker='o')
plt.axhline(y=black_scholes_call(S, K, T, r, sigma), color='r', linestyle='--', label="Black Scholes Option Price")

# Draw vertical lines for confidence intervals
for i, num_paths in enumerate(num_paths_list):
    plt.vlines(num_paths, confidence_intervals[0, i], confidence_intervals[1, i], colors='blue', linestyles='dashed', alpha=0.7, label=f"{0.95*100}% CI")

plt.title("EQ 2 Monte Carlo Option Price vs. Number of Paths with 95% Confidence Intervals")
plt.xlabel("Number of Paths")
plt.ylabel("Approx Call Option Price")
# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
plt.show()