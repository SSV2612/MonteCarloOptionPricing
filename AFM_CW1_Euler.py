import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from mrg32k3a.mrg32k3a import MRG32k3a

# Record the start time
# start_time = time.time()

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

def monte_carlo_simulation(S, K, T, r, sigma, num_paths, num_steps):
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
    average_option_price: Average option price using Monte Carlo simulation
    option_prices: Array of estimated call option prices for each path
    """
    dt = T / num_steps

    option_prices = np.zeros(num_paths)

    for i in range(num_paths):
        X_path = np.zeros(num_steps + 1)
        X_path[0] = S

        for j in range(1, num_steps + 1):
            # standard_normal_array = np.random.normal(0, 1, 1)
            standard_normal_array = rng.normalvariate(mu=0, sigma=1)
            X_path[j] = X_path[j - 1] * ( 1 + (r * dt) + (sigma * np.sqrt(dt) * standard_normal_array))

        # Calculate the option price at maturity (T)
        option_prices[i] = max(0, X_path[-1] - K) * np.exp(-r * T)

    # Calculate the average of all call option prices
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

def calculate_confidence_interval2(data):
    """
    Calculate confidence interval for the given data.
    """
    mean_value = np.mean(data)
    b_m = np.std(data)
    m = len(data)
    lower_bound = mean_value - (1.96 * b_m)/ np.sqrt(m)
    upper_bound = mean_value + (1.96 * b_m)/ np.sqrt(m)
    return lower_bound, upper_bound


# Example usage
S = 100  # Stock price S_{0}
K = 110  # Strike
T = 2  # Time to maturity
r = 0.05  # Risk-free rate in annual %
sigma = 0.25  # Annual volatility in %
num_steps = 2  # Time steps
num_paths = 20000000  # Number of trials

# Vary the number of Steps 
num_steps_list = [3, 4, 6, 8, 10]
option_prices_list = []
confidence_intervals = []
absolute_error_list = []
delta_list = []

monte, monte_prices = monte_carlo_simulation(S, K, T, r, sigma, num_paths, num_steps)
print(f"Monte Carlo Option Price is {monte}")
bs = black_scholes_call(S, K, T, r, sigma)
print(f"Black Scholes Option Price is {bs}")

# Calculate errors
abs_error = abs(monte - bs)
relative_error = abs(monte - bs) / bs
print("Absolute Error:", abs_error)
print("Relative Error:", relative_error)

for num_steps in num_steps_list:
    average_monte_price, monte_prices = monte_carlo_simulation(S, K, T, r, sigma, num_paths, num_steps)
    option_prices_list.append(average_monte_price)
    absolute_error = abs(average_monte_price - bs)
    absolute_error_list.append(absolute_error)
    delta_list.append(T/num_steps)

    # Calculate and store 95% confidence intervals
    lower_bound, upper_bound = calculate_confidence_interval(monte_prices)
    # lower_bound, upper_bound = calculate_confidence_interval2(monte_prices)
    confidence_intervals.append((abs(lower_bound - bs), abs(upper_bound - bs)))
    print(f"95% Confidence Interval for delta = {T/num_steps} is ({lower_bound}, {upper_bound})")
    print(f"95% Confidence Interval Difference =  ({upper_bound - lower_bound})")


# Plot the results with vertical lines for confidence intervals
option_prices_list = np.array(option_prices_list)
confidence_intervals = np.array(confidence_intervals).T
absolute_error_list = np.array(absolute_error_list)
print(f"Absolute Error list {delta_list} ")
print(f"Absolute Error list {absolute_error_list} ")


plt.figure(figsize=(10, 6))
plt.plot(np.log(delta_list), np.log(absolute_error_list), label="Monte Carlo Option Price", marker='o')


# Draw vertical lines for confidence intervals
for i,num_steps in enumerate(num_steps_list):
    plt.vlines(np.log(T/num_steps), np.log(confidence_intervals[0, i]), np.log(confidence_intervals[1, i]), colors='red', linestyles='dashed', alpha=0.7, label=f"{0.95*100}% CI")

plt.title("Absolute Error of Option Price vs. Delta t with 95% Confidence Intervals")
plt.xlabel("Delta t (Ln)")
plt.ylabel("Absolute Error (Ln)")
plt.show()