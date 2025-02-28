# 📘 Monte Carlo Simulations for Option Pricing

This project applies Monte Carlo simulation to estimate prices for European call options, focusing on plain vanilla and binary cash-or-nothing payoffs. Given Monte Carlo's slow convergence, various variance reduction techniques are explored to improve accuracy and efficiency.

🔹 Key Features

-Monte Carlo Simulation for option pricing.
-Antithetic Variates to enhance convergence and accuracy.
-Weak-Euler Discretization to improve SDE approximations.
-Analysis of convergence speed for different techniques.

Through a series of simulations, these methods were shown to improve precision and computational efficiency. This repository contains Python implementations of these techniques, demonstrating their impact on Monte Carlo-based pricing models.

Two Discretisation methods were used to calculate the Call option prices using Monte Carlo Simulation -

- **Euler-Maruyama**(First Discretisation)
- **Geometric Brownian Motion**(Second Discretisation)

## 🔧 Getting Started

### 1️⃣ Prerequisites

You’ll need **Python 3.x** installed. `pipenv` was used for easy dependency management.

### 2️⃣ Set Up Your Virtual Environment

In your project folder, run:

```bash
pipenv shell
```

### 3️⃣ Install Dependencies

To install all required libraries:

```bash
pipenv install numpy scipy matplotlib mrg32k3a
```

### 4️⃣ Verify Everything is Set Up

```bash
pipenv graph
```

### 5️⃣ Run the Simulations

Each Python script explores different simulation techniques. Run them like this:

```bash
pipenv run python AFM_AntitheticVariate_BinaryCash.py
pipenv run python AFM_AntitheticVariate_Call.py
pipenv run python AFM_ControlVariate_BinaryCash.py
pipenv run python AFM_ControlVariate_Call.py
pipenv run python AFM_CW1_AntiPaths.py
pipenv run python AFM_Cw1_Cash.py
pipenv run python AFM_CW1_Euler.py
pipenv run python AFM_Cw1_SDE2_Csh.py
pipenv run python AFM_Cw1_SDE2.py
```

## 📜 What Each File Does

- **AFM_AntitheticVariate_BinaryCash.py**: Uses **Antithetic Variate** to improve Monte Carlo accuracy for **Binary Cash-or-Nothing Options**.
- **AFM_AntitheticVariate_Call.py**: Similar, but for Vanilla call options.
- **AFM_ControlVariate_BinaryCash.py**: Uses Control Variate to reduce variance in pricing Binary Cash-or-Nothing options.
- **AFM_ControlVariate_Call.py**: Same as above, but for Vanilla Call options.
- **AFM_CW1_AntiPaths.py**: Generates Antithetic Paths for variance reduction in simulations.
- **AFM_Cw1_Cash.py**: Implements Euler-Maruyama method for Binary Cash-or-Nothing Options and compares Monte Carlo Error vs Euler Error.
- **AFM_CW1_Euler.py**: Plots Euler discretization errors for the first discretization method.
- **AFM CW1.py** - Plots Monte Carlo error for Euler discretization.
- **AFM_Cw1_SDE2_Csh.py**: Similar to AFM_Cw1_Cash.py but with second discretisation(GBM).
- **AFM_Cw1_SDE2.py**: A Vanilla Call Option pricing model using first discretisation(GBM).
- **QMC_binarycash.py**: Binary Cash-or-Nothing call option pricing using Quasi Monte Carlo Technique.
- **QMC_calloption.py**: Plain Vanilla Call Option pricing using Quasi Monte Carlo Technique.

## List of Input Parameters to be played with

- **S** - Initial stock price
- **K** - Strike price
- **T** - Time to Maturity
- **r** - Risk free rate in annual %
- **sigma** - Annual Volatility in %
- **num_steps** - time steps
- **num_paths** - number of trials

This repository provides a structured and efficient implementation of Monte Carlo-based option pricing, leveraging variance reduction techniques and discretization methods to improve convergence and accuracy. It serves as a valuable resource for financial modeling and quantitative analysis.
