import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import qmc
from scipy.special import erfinv
from mrg32k3a.mrg32k3a import MRG32k3a

# Initialisation of the RNG object using L’Ecuyer’s MRG --  MRG32k3a
rng = MRG32k3a(ref_seed=(12345,54321,1388,54321,12345,99211))

# Caclulation of exact option price for plain vanilla call 
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

# Calculation of the exact option price for binary cash-or-nothing
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

def monte_carlo_binary_cash_or_nothing(S, K, T, r, sigma, num_paths, num_steps, cash_amount=1):
    """
    Inputs:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility
    num_paths: Number of Monte Carlo paths to simulate
    num_steps: Number of time steps
    cash_amount: Fixed cash amount if the option is in the money
    
    Output:
    binary_cash_prices: Array of binary cash-or-nothing option prices for each path
    """
    dt = T / num_steps

    binary_cash_prices = np.zeros(num_paths)

    for i in range(num_paths):
        X_path = np.zeros(num_steps + 1)
        X_path[0] = S

        for j in range(1, num_steps + 1):
            #sample random numbers from L’Ecuyer’s MRG using in built package --  MRG32k3a
            standard_normal_array = rng.normalvariate(mu=0,sigma=1)
            X_path[j] = X_path[j - 1] * np.exp((r-0.5*sigma**2) * dt + sigma * np.sqrt(dt) * standard_normal_array)

        # Check if the option is in the money at maturity (T)
        if X_path[-1] >= K:
            binary_cash_prices[i] = cash_amount * np.exp(-r * T)
        else:
            binary_cash_prices[i] = 0
    
    #return binary cash price from each path
    return binary_cash_prices
    

# Call option price estimation using control variate (binary cash) 
def controlvariate_call(S, K, T, r, sigma, num_paths, num_steps,binary_cash_pricelist):
    dt = T/num_steps
    option_prices = np.zeros(num_paths)

    for i in range(num_paths):
        S_Path = np.zeros(num_steps+1)
        S_Path[0] = S
        for j in range(1,num_steps+1):
            #sample random numbers from L’Ecuyer’s MRG using in built package --  MRG32k3a
            standard_normal_array = rng.normalvariate(mu=0,sigma=1)
            S_Path[j] = S_Path[j - 1] * np.exp((r-0.5*sigma**2) * dt + sigma * np.sqrt(dt) * standard_normal_array)
        
        option_prices[i] = (max(0,S_Path[-1]-K) * np.exp(-r * T))
    
    # Covariance of call and binary cash option price 
    covariance = np.mean(option_prices*binary_cash_pricelist) - (np.mean(option_prices)*np.mean(binary_cash_pricelist))

    # Coefficient calculation which determines the correlation between the two options
    theta = covariance/np.var(binary_cash_pricelist)

    # Control Variate -- Call option price calculation using binary cash option as control variate
    Z = option_prices + theta*(np.mean(binary_cash_pricelist)-binary_cash_pricelist)
    controlvariate_option_prices = np.mean(Z)

    #return plain vanilla call option price for each path and the estimated plain vanilla call option price
    return controlvariate_option_prices, Z    

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for the given data.
    """
    mean_value = np.mean(data)
    margin_of_error = norm.ppf(1 - (1 - confidence) / 2) * np.std(data) / np.sqrt(len(data))
    lower_bound = mean_value - margin_of_error
    upper_bound = mean_value + margin_of_error
    return lower_bound, upper_bound


S = 100 # stock price S_{0}
K = 110  # strike
T = 2  # time to maturity
r = 0.05  # risk-free rate in annual %
sigma = 0.25  # annual volatility in %
num_steps = 100  # time steps
num_paths = 10**5  # initial number of trials

option_prices_list = []
confidence_intervals = []
absolute_error_list = []


binary_cashprice_list = monte_carlo_binary_cash_or_nothing(S, K, T, r, sigma, num_paths, num_steps)
monte, monte_prices = controlvariate_call(S, K, T, r, sigma, num_paths, num_steps, binary_cashprice_list)
print(f"SDE2 Monte Carlo Option Price is {monte}")
bs = black_scholes_call(S, K, T, r, sigma)
print(f"SDE2 Black Scholes Option Price is {bs}")

# # Calculate errors
absolute_error = abs(monte - bs)
relative_error = abs(monte - bs) / bs
print("Absolute Error:", absolute_error)
print("Relative Error:", relative_error)


num_paths_list = [400, 1000, 1500, 2000, 3000, 4500, 6000, 9000, 10000]

# can modify the num_paths in below list and check for various number of paths
#num_paths_list = [10000, 25000, 50000, 75000, 10**5]

for num_paths in num_paths_list:
    binary_cashprice_list = monte_carlo_binary_cash_or_nothing(S, K, T, r, sigma, num_paths, num_steps) #Calculate binary cash option price for each path using Weak Euler method
    start_time = time.time()
    average_monte_price, monte_prices = controlvariate_call(S, K, T, r, sigma, num_paths, num_steps, binary_cashprice_list) #Call option price calculation using Control Variate
    end_time = time.time()
    option_prices_list.append(average_monte_price)
    absolute_error = abs(average_monte_price - bs)
    absolute_error_list.append(absolute_error)

    lower_bound, upper_bound = calculate_confidence_interval(monte_prices)
    confidence_intervals.append((lower_bound, upper_bound))
    confidence_interval_width = upper_bound - lower_bound
    print(f"95% Confidence Interval for {num_paths} paths: ({lower_bound}, {upper_bound})")
    print(f"95% C.I width for {num_paths}:{confidence_interval_width}")

print(option_prices_list)
final_abs_error = np.mean(absolute_error_list)
print("Average Absolute Error:",final_abs_error)
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

plt.title("Monte Carlo Call Option Price vs. Number of Paths with 95% Confidence Intervals using Control Variate")
plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.show()