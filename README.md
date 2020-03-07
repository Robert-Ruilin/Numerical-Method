# Numerical-Method
Option Pricing Algorithm and Monte Carlo Simulation

1. Greeks Simulation
Greeks measure the sensitivity of the price of derivatives to a change in underlying asset’s parameters. They are used for hedging and risk management. The commonly used greeks are:
Delta: measures the rate of change of the option value with respect to changes in the underlying asset's price.
Gamma: measures the rate of change of delta with respect to changes in the underlying asset's price.
Vega: measures the rate of change of the option value with respect to changes in the underlying asset's volatility.
Theta: the rate of change in the price of an option with respect to time.
Rho (PV01): the rate of change in the price of an option in response to a change in the interest rate.

Implement a Greeks calculater for binomial tree pricer provided below. The signature of the greeks calculator is:

def binomialGreeks(S, r, vol, T, strike, greekType) -> float

Setting 𝑆=100,𝑟=0.03,𝑣𝑜𝑙=0.2,𝑇=1, plot each greeks as a function of strike from 50 to 150. Play with different binomial models and see if there is any difference.
