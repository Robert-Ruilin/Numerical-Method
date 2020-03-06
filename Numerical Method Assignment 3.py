# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:20:52 2020

@author: LENOVO
"""
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import math

class PayoffType(Enum):
    Call = 0
    Put = 1

class EuropeanOption():
    def __init__(self, expiry, strike, payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self, S):
        if self.payoffType == PayoffType.Call:
            return max(S - self.strike, 0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike - S, 0)
        else:
            raise Exception("payoffType not supported: ", self.payoffType)
    def valueAtNode(self, t, S, continuation):
        return continuation

class AmericanOption():
    def __init__(self, expiry, strike, payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self, S):
        if self.payoffType == PayoffType.Call:
            return max(S - self.strike, 0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike - S, 0)
        else:
            raise Exception("payoffType not supported: ", self.payoffType)
    def valueAtNode(self, t, S, continuation):
        return max(self.payoff(S), continuation)

def crrCalib(r, vol, t):
    b = math.exp(vol * vol * t + r * t) + math.exp(-r * t)
    u = (b + math.sqrt(b * b - 4)) / 2
    p = (math.exp(r * t) - (1 / u)) / (u - 1 / u)
    return (u, 1/u, p)

def jrrnCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

def jreqCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    return (u, d, 1/2)

def tianCalib(r, vol, t):
    v = math.exp(vol * vol * t)
    u = 0.5 * math.exp(r * t) * v * (v + 1 + math.sqrt(v*v + 2*v - 3))
    d = 0.5 * math.exp(r * t) * v * (v + 1 - math.sqrt(v*v + 2*v - 3))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

def binomialPricer(S, r, vol, trade, n, calib):
    t = trade.expiry / n
    (u, d, p) = calib(r, vol, t)
    # set up the last time slice, there are n+1 nodes at the last time slice
    vs = [trade.payoff(S * u ** (n - i) * d ** i) for i in range(n + 1)]
    # iterate backward
    for i in range(n - 1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i + 1):
            nodeS = S * u ** (i - j) * d ** j
            continuation = math.exp(-r * t) * (vs[j] * p + vs[j + 1] * (1 - p))
            vs[j] = trade.valueAtNode(t * i, nodeS, continuation)
    return vs[0]

def binomialGreeks(S, r, vol, T, strike, greekType) -> float:
    s, sigma, ts = 1, 0.002, 1/252
    expiry = T
    k = range(50,151)
    greekletter_EuroCall = np.zeros(shape = len(k))
    greekletter_EuroPut = np.zeros(shape = len(k))
    greekletter_AmerCall = np.zeros(shape = len(k))
    greekletter_AmerPut = np.zeros(shape = len(k))
    if greekType == 1:
#delta - crrcalibration
        calib = crrCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        fig, ax1 = plt.subplots(figsize = (10,7))
        ax1.plot(k, greekletter_EuroCall, 'b', label = 'delta for European Call Option')
        ax1.plot(k, greekletter_EuroPut, 'k', label = 'delta for European Put Option')
        ax1.plot(k, greekletter_AmerCall, 'g', label = 'delta for American Call Option')
        ax1.plot(k, greekletter_AmerPut, 'y', label = 'delta for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('delta')
        plt.xlabel('strike price')
        ax1.set_title('delta calibrated by crr')
        ax1.legend()
        plt.show()
#delta - jrrncalibration
        calib = jrrnCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        fig, ax2 = plt.subplots(figsize = (10,7))
        ax2.plot(k, greekletter_EuroCall, 'b', label = 'delta for European Call Option')
        ax2.plot(k, greekletter_EuroPut, 'k', label = 'delta for European Put Option')
        ax2.plot(k, greekletter_AmerCall, 'g', label = 'delta for American Call Option')
        ax2.plot(k, greekletter_AmerPut, 'y', label = 'delta for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('delta')
        plt.xlabel('strike price')
        ax2.set_title('delta calibrated by jrrn')
        ax2.legend()
        plt.show()
#delta - jreqcalibration
        calib = jreqCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        fig, ax3 = plt.subplots(figsize = (10,7))
        ax3.plot(k, greekletter_EuroCall, 'b', label = 'delta for European Call Option')
        ax3.plot(k, greekletter_EuroPut, 'k', label = 'delta for European Put Option')
        ax3.plot(k, greekletter_AmerCall, 'g', label = 'delta for American Call Option')
        ax3.plot(k, greekletter_AmerPut, 'y', label = 'delta for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('delta')
        plt.xlabel('strike price')
        ax3.set_title('delta calibrated by jreq')
        ax3.legend()
        plt.show()
#delta - tiancalibration
        calib = tianCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-binomialPricer(S-s,r,vol,trade,n,calib))/(2*s)
        fig, ax4 = plt.subplots(figsize = (10,7))
        ax4.plot(k, greekletter_EuroCall, 'b', label = 'delta for European Call Option')
        ax4.plot(k, greekletter_EuroPut, 'k', label = 'delta for European Put Option')
        ax4.plot(k, greekletter_AmerCall, 'g', label = 'delta for American Call Option')
        ax4.plot(k, greekletter_AmerPut, 'y', label = 'delta for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('delta')
        plt.xlabel('strike price')
        ax4.set_title('delta calibrated by tian')
        ax4.legend()
        plt.show()
#gamma - crrcalibration
    if greekType == 2:
        calib = crrCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        fig, ax5 = plt.subplots(figsize = (10,7))
        ax5.plot(k, greekletter_EuroCall, 'b', label = 'gamma for European Call Option')
        ax5.plot(k, greekletter_EuroPut, 'k', label = 'gamma for European Put Option')
        ax5.plot(k, greekletter_AmerCall, 'g', label = 'gamma for American Call Option')
        ax5.plot(k, greekletter_AmerPut, 'y', label = 'gamma for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('gamma')
        plt.xlabel('strike price')
        ax5.set_title('gamma calibrated by crr')
        ax5.legend()
        plt.show()
#gamma - jrrncalibration
        calib = jrrnCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        fig, ax6 = plt.subplots(figsize = (10,7))
        ax6.plot(k, greekletter_EuroCall, 'b', label = 'gamma for European Call Option')
        ax6.plot(k, greekletter_EuroPut, 'k', label = 'gamma for European Put Option')
        ax6.plot(k, greekletter_AmerCall, 'g', label = 'gamma for American Call Option')
        ax6.plot(k, greekletter_AmerPut, 'y', label = 'gamma for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('gamma')
        plt.xlabel('strike price')
        ax6.set_title('gamma calibrated by jrrn')
        ax6.legend()
        plt.show()
#gamma - jreqcalibration
        calib = jreqCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        fig, ax7 = plt.subplots(figsize = (10,7))
        ax7.plot(k, greekletter_EuroCall, 'b', label = 'gamma for European Call Option')
        ax7.plot(k, greekletter_EuroPut, 'k', label = 'gamma for European Put Option')
        ax7.plot(k, greekletter_AmerCall, 'g', label = 'gamma for American Call Option')
        ax7.plot(k, greekletter_AmerPut, 'y', label = 'gamma for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('gamma')
        plt.xlabel('strike price')
        ax7.set_title('gamma calibrated by jreq')
        ax7.legend()
        plt.show()
#gamma - tiancalibration
        calib = tianCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S+s,r,vol,trade,n,calib)-2*binomialPricer(S,r,vol,trade,n,calib)+binomialPricer(S-s,r,vol,trade,n,calib))/(s**2)
        fig, ax8 = plt.subplots(figsize = (10,7))
        ax8.plot(k, greekletter_EuroCall, 'b', label = 'gamma for European Call Option')
        ax8.plot(k, greekletter_EuroPut, 'k', label = 'gamma for European Put Option')
        ax8.plot(k, greekletter_AmerCall, 'g', label = 'gamma for American Call Option')
        ax8.plot(k, greekletter_AmerPut, 'y', label = 'gamma for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('gamma')
        plt.xlabel('strike price')
        ax8.set_title('gamma calibrated by tian')
        ax8.legend()
        plt.show()
#vega - crrcalibration
    if greekType == 3:
        calib = crrCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        fig, ax9 = plt.subplots(figsize = (10,7))
        ax9.plot(k, greekletter_EuroCall, 'b', label = 'vega for European Call Option')
        ax9.plot(k, greekletter_EuroPut, 'k', label = 'vega for European Put Option')
        ax9.plot(k, greekletter_AmerCall, 'g', label = 'vega for American Call Option')
        ax9.plot(k, greekletter_AmerPut, 'y', label = 'vega for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('vega')
        plt.xlabel('strike price')
        ax9.set_title('vega calibrated by crr')
        ax9.legend()
        plt.show()
#vega - jrrncalibration
        calib = jrrnCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        fig, ax10 = plt.subplots(figsize = (10,7))
        ax10.plot(k, greekletter_EuroCall, 'b', label = 'vega for European Call Option')
        ax10.plot(k, greekletter_EuroPut, 'k', label = 'vega for European Put Option')
        ax10.plot(k, greekletter_AmerCall, 'g', label = 'vega for American Call Option')
        ax10.plot(k, greekletter_AmerPut, 'y', label = 'vega for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('vega')
        plt.xlabel('strike price')
        ax10.set_title('vega calibrated by jrrn')
        ax10.legend()
        plt.show()
#vega - jreqcalibration
        calib = jreqCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        fig, ax11 = plt.subplots(figsize = (10,7))
        ax11.plot(k, greekletter_EuroCall, 'b', label = 'vega for European Call Option')
        ax11.plot(k, greekletter_EuroPut, 'k', label = 'vega for European Put Option')
        ax11.plot(k, greekletter_AmerCall, 'g', label = 'vega for American Call Option')
        ax11.plot(k, greekletter_AmerPut, 'y', label = 'vega for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('vega')
        plt.xlabel('strike price')
        ax11.set_title('vega calibrated by jreq')
        ax11.legend()
        plt.show()
#vega - tiancalibration
        calib = tianCalib
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        for i in range(len(k)):
            strike = k[i]
            trade=AmericanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S,r,vol+sigma,trade,n,calib)-binomialPricer(S,r,vol-sigma,trade,n,calib))/(2*sigma)
        fig, ax12 = plt.subplots(figsize = (10,7))
        ax12.plot(k, greekletter_EuroCall, 'b', label = 'vega for European Call Option')
        ax12.plot(k, greekletter_EuroPut, 'k', label = 'vega for European Put Option')
        ax12.plot(k, greekletter_AmerCall, 'g', label = 'vega for American Call Option')
        ax12.plot(k, greekletter_AmerPut, 'y', label = 'vega for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('vega')
        plt.xlabel('strike price')
        ax12.set_title('vega calibrated by tian')
        ax12.legend()
        plt.show()
#theta - crrcalibration
    if greekType == 4:
        calib = crrCalib
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Call)
            trade2=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Put)
            trade2=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=AmericanOption(expiry+ts,strike,PayoffType.Call)
            trade2=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Put)
            trade2=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        fig, ax13 = plt.subplots(figsize = (10,7))
        ax13.plot(k, greekletter_EuroCall, 'b', label = 'theta for European Call Option')
        ax13.plot(k, greekletter_EuroPut, 'k', label = 'theta for European Put Option')
        ax13.plot(k, greekletter_AmerCall, 'g', label = 'theta for American Call Option')
        ax13.plot(k, greekletter_AmerPut, 'y', label = 'theta for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('theta')
        plt.xlabel('strike price')
        ax13.set_title('theta calibrated by crr')
        ax13.legend()
        plt.show()
#theta - jrrncalibration
        calib = jrrnCalib
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Call)
            trade2=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Put)
            trade2=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=AmericanOption(expiry+ts,strike,PayoffType.Call)
            trade2=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Put)
            trade2=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        fig, ax14 = plt.subplots(figsize = (10,7))
        ax14.plot(k, greekletter_EuroCall, 'b', label = 'theta for European Call Option')
        ax14.plot(k, greekletter_EuroPut, 'k', label = 'theta for European Put Option')
        ax14.plot(k, greekletter_AmerCall, 'g', label = 'theta for American Call Option')
        ax14.plot(k, greekletter_AmerPut, 'y', label = 'theta for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('theta')
        plt.xlabel('strike price')
        ax14.set_title('theta calibrated by jrrn')
        ax14.legend()
        plt.show()
#theta - jreqcalibration
        calib = jreqCalib
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Call)
            trade2=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Put)
            trade2=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=AmericanOption(expiry+ts,strike,PayoffType.Call)
            trade2=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Put)
            trade2=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        fig, ax15 = plt.subplots(figsize = (10,7))
        ax15.plot(k, greekletter_EuroCall, 'b', label = 'theta for European Call Option')
        ax15.plot(k, greekletter_EuroPut, 'k', label = 'theta for European Put Option')
        ax15.plot(k, greekletter_AmerCall, 'g', label = 'theta for American Call Option')
        ax15.plot(k, greekletter_AmerPut, 'y', label = 'theta for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('theta')
        plt.xlabel('strike price')
        ax15.set_title('theta calibrated by jreq')
        ax15.legend()
        plt.show()
#theta - tiancalibration
        calib = tianCalib
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Call)
            trade2=EuropeanOption(expiry,strike,PayoffType.Call)
            greekletter_EuroCall[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Put)
            trade2=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_EuroPut[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=AmericanOption(expiry+ts,strike,PayoffType.Call)
            trade2=AmericanOption(expiry,strike,PayoffType.Call)
            greekletter_AmerCall[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        for i in range(len(k)):
            strike = k[i]
            trade1=EuropeanOption(expiry+ts,strike,PayoffType.Put)
            trade2=EuropeanOption(expiry,strike,PayoffType.Put)
            greekletter_AmerPut[i] = (binomialPricer(S,r,vol,trade1,n,calib)-binomialPricer(S,r,vol,trade2,n,calib))/(ts)
        fig, ax16 = plt.subplots(figsize = (10,7))
        ax16.plot(k, greekletter_EuroCall, 'b', label = 'theta for European Call Option')
        ax16.plot(k, greekletter_EuroPut, 'k', label = 'theta for European Put Option')
        ax16.plot(k, greekletter_AmerCall, 'g', label = 'theta for American Call Option')
        ax16.plot(k, greekletter_AmerPut, 'y', label = 'theta for American Put Option')
        plt.xlim((50,150))
        plt.xticks(np.arange(50,150,20))
        plt.ylabel('theta')
        plt.xlabel('strike price')
        ax16.set_title('theta calibrated by tian')
        ax16.legend()
        plt.show()


# function input
S, r, vol, T, n = 100, 0.03, 0.2, 1, 252
strike = np.linspace(50,150,101)

# function application
binomialGreeks(S,r,vol,T,strike,4)





