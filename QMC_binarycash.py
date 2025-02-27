import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import qmc
from scipy.special import erfinv

# Binary cash-or-nothing exact option price calculation
def black_scholes_binary_cash_or_nothing(S, K, T, r, sigma, cash_amount=1):
    """
    Inputs
    S: Current stock Price
    K: Strike Price
    T: Time to maturity (1 year = 1, 1 month = 1/12)
    r: Risk-free interest rate
    sigma: Volatility
    cash_amount: Fixed cash amount if the option is in the money
    
    Output
    binary_cash_price: Value of the binary cash-or-nothing option 
    """
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        raise ValueError("Invalid input values. Ensure positive values for T, S, K, and sigma.")

    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    binary_cash_price = cash_amount * np.exp(-r * T) * norm.cdf(d2)
    return binary_cash_price

# Quasi monte carlo implementation for binary cash-or-nothing
def quasimontecarlo_binarycash(S, K, T, r, sigma, num_paths, num_steps, cash_amount=1):
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
   binary_cash_prices = np.zeros(num_paths)

   for i in range(num_paths):
      #Construct paths
      S_Path = np.zeros(num_steps+1)

      #Set initial stock price at time 0 for each path
      S_Path[0] = S

      for j in range(1,num_steps+1):
         sampler = qmc.Halton(d=1,scramble=True)
         halton_paths = sampler.random(1)
         norm_halton_paths = np.sqrt(2)*erfinv((2 * halton_paths) - 1)
         S_Path[j] = S_Path[j - 1] * np.exp((r-0.5*sigma**2) * dt + sigma * np.sqrt(dt) * norm_halton_paths)

         if S_Path[-1] >= K:
            binary_cash_prices[i] = cash_amount * np.exp(-r * T)
         else:
            binary_cash_prices[i] = 0
      
      #Option price calculation at maturity
      average_binary_cash_price = np.mean(binary_cash_prices)

   return average_binary_cash_price, binary_cash_prices

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

monte_binary_cash, binary_cash = quasimontecarlo_binarycash(S, K, T, r, sigma, num_paths, num_steps)
bs_binary_cash = black_scholes_binary_cash_or_nothing(S, K, T, r, sigma)

print("Monte Carlo Price (Binary Cash-or-Nothing):", monte_binary_cash)
print("Black-Scholes Price (Binary Cash-or-Nothing):", bs_binary_cash)

# Calculate errors
absolute_error_binary_cash = abs(monte_binary_cash - bs_binary_cash)
relative_error_binary_cash = abs(monte_binary_cash - bs_binary_cash) / bs_binary_cash
print("Absolute Error:", absolute_error_binary_cash)
print("Relative Error:", relative_error_binary_cash)

for num_paths in num_paths_list:
    average_monte_price, monte_prices = quasimontecarlo_binarycash(S, K, T, r, sigma, num_paths, num_steps)
    option_prices_list.append(average_monte_price)
    absolute_error = abs(average_monte_price - bs_binary_cash)
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
plt.axhline(y=black_scholes_binary_cash_or_nothing(S, K, T, r, sigma), color='r', linestyle='--', label="Black Scholes Option Price")

# Draw vertical lines for confidence intervals
for i, num_paths in enumerate(num_paths_list):
    plt.vlines(num_paths, confidence_intervals[0, i], confidence_intervals[1, i], colors='blue', linestyles='dashed', alpha=0.7, label=f"{0.95*100}% CI")

plt.title("Quasi Monte Carlo Option Price vs. Number of Paths with 95% Confidence Intervals")
plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.show()