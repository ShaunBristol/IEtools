import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import itertools

def gie(X, k, a):
    return a*X**k

def gie_log(X, k, a):
    return k*np.log(X)+a

def gie_loss(x, X, y):
    return np.sum(np.abs(y - gie(X, x[0], x[1])))

def gie_log_loss(x, X, y):
    return np.sum(np.abs(np.log(y) - gie_log(X, x[0], x[1])))

class GeneralIE:
    def __init__(self, k=None, a=None, log=True):
        self.k = k
        self.a = a
        self.log = log
        
    def predict(self, X):
        if not self.k or not self.a: raise ValueError
        
        gie_func = gie_log if self.log is True else gie
        prediction = gie_func(X, self.k, self.a)
        if self.log: prediction = np.exp(prediction)
        return prediction
    
    def fit(self, X, y, guess = np.array((1.0,0.0)), log=None):
        assert(X.shape[0] == y.shape[0])

        if log is not None: self.log = log
        
        gie_func = gie_log if self.log is True else gie
        
        ppot, pcov = curve_fit(gie_func, X, np.log(y) if self.log is True else y, p0=guess)
        #perr = np.sqrt(np.diag(pcov))
        self.k, self.a = ppot
        return self

#Relative Time
#Returns the elapsed time between the submitted date, X, and the base time, in years as a floating point
#Date format must be compatible with pandas.to_datetime
class RelTime:
    def __init__(self, base = "1900-1-1", scale = 1.0, params = None):
        self.base = pd.to_datetime(base, **(params or {}))
        self.scale = scale
    
    def transform(self, X, params = None):
        X = pd.to_datetime(X, **(params or {}))
        result = (X - self.base)//pd.Timedelta(1, unit='d') * self.scale / 365.25
        return result
        
    def inverse_transform(self, X, start = None):
        return self.base + pd.to_timedelta(X*365.25/self.scale ,'d') 
        
class LogLinearEntropyMin:
    
    def __init__(self, alpha = None):
        self.alpha = alpha
        self.y = None
        
    def fit(self, X, y, method="brute", range = None, alpha_range=(-0.1,0.1), alpha_delta=0.1, n_bins=40):
        self.y =  y
        
        if range is None:
            range = np.linspace(alpha_range[0], alpha_range[1], num=round(1/alpha_delta))
            
        if method is "brute":
            result_list = list()
            for alpha in range:
                log_linear = np.histogram(self.log_linear(X, self.y, alpha), bins=n_bins)[0]
                result_list.append(stats.entropy(log_linear, base=log_linear.shape[0]))
            self.alpha = range[np.array(result_list).argmin()]
        else:
            raise ValueError
        return self
        
    def transform(self, X):
        return self.log_linear(X, self.y, self.alpha)
    
    @staticmethod
    def log_linear(X, y, alpha):
        return np.log(X) - alpha*(y)
    
    def fit_transform(self, X, y, method="brute", range = None, alpha_range=(-0.1,0.1), alpha_delta=0.01, n_bins=25):
        self.fit(X, y, method, range, alpha_range, alpha_delta, n_bins)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return np.exp(X + self.alpha*(self.y))
        
def shock(x, c, *args):
    if np.mod(len(args),3) != 0:
        raise ValueError
    elif len(args) is 0:
        return c
    else:
        a = args[0]
        b = args[1]
        t = args[2]
        result = a/(1 + np.exp((x-t)/b))
        return  result + shock(x, c, *args[3:])

def shock_eq(x, c, alpha, *args):
    if np.mod(len(args),3) != 0:
        raise ValueError
    elif len(args) is 0:
        return alpha*x+c
    else:
        a = args[0]
        b = args[1]
        t = args[2]
        result = a/(1 + np.exp((x-t)/b))
        return result + shock_eq(x, c, alpha, *args[3:])
        
#X is independent variable, is time variable
#y is the dependent variable, transformed time series

class DynamicIE:
    def __init__(self, eq=False):
        self.min_date = None
        self.results = None
        self.pcov = None
        self.ds = None
        self.eq = eq
    
    def fit(self, X, y, guesses=None, guess_c=0.0, guess_alpha = 1.0, eq=None, n_shocks=None, bound=True):
        self.eq = eq
        #np.seterr(all='raise')
        if guesses is not None:
            n_shocks = len(guesses)
        
        func = shock
        guess = [guess_c]
        if self.eq is True:
            func = shock_eq
            guess = guess + [guess_alpha]
        
        bounds = (-np.inf, np.inf)
        if bound is True:
            min_bounds = [-np.inf]
            max_bounds = [np.inf]
            if eq is True:
                min_bounds.append(-np.inf)
                max_bounds.append(np.inf)
            for _ in range(n_shocks):
                min_bounds.append(-np.inf)
                max_bounds.append(np.inf)
                min_bounds.append(-np.inf)
                max_bounds.append(np.inf)
                min_bounds.append(min(X))
                max_bounds.append(max(X))
            bounds = (min_bounds, max_bounds)
                
        if guesses is None and n_shocks is not None:
            guess = guess + np.full((n_shocks*3,), 1.0).tolist()
        elif guesses is not None:
            #Flatten guesses
            guess = guess + list(itertools.chain.from_iterable(guesses))
        else: raise ValueError
        
        self.popt, self.pcov = curve_fit(func, X, np.log(y), p0=guess, bounds=bounds, **{"verbose" : 0})
        
        it = iter(self.popt)
        
        self.results = {"c":next(it), "magnitude":list(),"width":list(),"transition":list()}
        if self.eq is True:
            self.results['alpha'] = next(it)
        
        for i in it:
            self.results['magnitude'].append(i)
            self.results['width'].append(next(it))
            self.results["transition"].append(next(it))
        
        return self
    
    def predict(self, X, y):
        #y = np.log(y)
        func = shock_eq if self.eq is True else shock
        return np.exp(func(X, *self.popt))