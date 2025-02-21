
import numpy as np
import math


def window(Length:int,Type: str = "Rectangular"):
    """windowing function in time domain

    Args:
        Length (int, optional): Sample length for the windowing. Defaults to 64.
        Type (str, optional): Type of window function wanted. Defaults to "Rectangular".
        Available Types: (Besides Rectangular)
        1- Hann
        2- Hamming
        3- Blackman
        4- Bartlett/triagunlar
        5- Welch
        6- Flat-Top Window

        
    """
    Type = Type.upper()
    if Type is "RECTANGULAR": output = np.ones(Length)
    elif Type is "HANN" or "HANNING": output = HANN(Length)
    elif Type is "HAMMING": output = HAMMING(Length)
    elif Type is "TRIANGULAR" or "BARTLETT" or "TRIANG": output = BARTLETT(Length)
    elif Type is "WELCH": output = WELCH_WINDOW(Length)
    elif Type is "FLAT TOP" or "FLATTOP" or "FLAT-TOP": output=FLAT_TOP_WINDOW(Length)
    return output

def RECTANGULAR(L): return np.tile(1,L)
def HANN(L): return COSINEWINDOW_K1(L,0.5)
def HAMMING(L): return COSINEWINDOW_K1(L,25/46)


def COSINEWINDOW_K1(L:int=64,a0:np.float64=0.5):
    """Generalised Cosine-sum window for K = 1
    form is a0 - (1-a0)cos(2pi*n/N)  0 =< n =< N

    Args:
        L (int, optional): Window Length. Defaults to 64.
        a0 (np.float64, optional): _description_. Defaults to 0.5.

    Returns:
        A K = 1 generalised cosine sum window: _description_
    """
    n = np.linspace(0,L-1,L)
    a0 = np.tile(a0,L)
    return a0 - (1-a0)*np.cos(2*math.pi*n/(L-1))

def GENERALISED_COSINE_SUM_WINDOW(L,K,a):
    pass

def WELCH_WINDOW(L: int):
    """Returns Welch Window of Length L

    Args:
        L (_type_): Length of Window

    """
    n = np.linspace(0,L-1,L)
    return np.ones(L)-((n-(L-1)/2)/((L-1)/2))**2

def BARTLETT(L: int):
    """Returns Triangular (Bartlett Variation) window of Length L

    Args:
        L int: Length of window 
    """
    n = np.linspace(0,L,L)
    N = np.tile(L,L)
    return np.ones(L)-np.abs((2*n-N)/L)

def FLAT_TOP_WINDOW(L: int):
    """Returns Flat top window based on coefficients from matlab flat top window

    Args:
        L (int): Window Length
    """
    a0 = np.tile(0.21557895,L)
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368
    n = np.linspace(0,L-1,L)
    N = np.tile(L-1,L)
    return a0 - a1*np.cos(2*math.pi*n/N)+a2*np.cos(4*math.pi*n/N)-a3*np.cos(6*math.pi*n/N)+a4*np.cos(8*math.pi*n/N)

