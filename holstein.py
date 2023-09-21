"""
Functions for calculating the RIXS cross-section, XAS and related quantitites for a Holstein phonon model.
"""

from numpy import *
from scipy.linalg import inv
from scipy.optimize import *

from builtins import sum #for bringing back functions shadowed by numpy
import operator as op
import itertools as it
from functools import *

from propagators import Gu, neighbor


@lru_cache(maxsize=None)
def An(j,z,t,U,w0,M):
    """
    Continued fraction all coefficients for all generalized Green's functions in the IMA0 approximation.
    Based on EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    nph = 4*max(int(ceil((M/w0)**2)), 1)
    A = array([(n, M*Gu(z-n*w0,t,U,j,j)) for n in range(nph,0,-1)])
    A0 = reduce(lambda x,y: product(y)/(1-y[1]*product(x)), A[1:])
    An = list(it.accumulate(A,
                            lambda x,y: product(y)/(1-y[1]*x),
                            initial=0.))
    A1 = An[-1]
    while abs((A1-A0)/A1)>1e-16:
        nph *= 2
        A = array([(n, M*Gu(z-n*w0,t,U,j,j)) for n in range(nph,0,-1)])
        An = list(it.accumulate(A,
                                lambda x,y: product(y)/(1-y[1]*x),
                                initial=0.))
        A0 = A1
        A1 = An[-1]
    return array(An[:0:-1])/sqrt(A[::-1,0]) #For Fn normalized by sqrt(n!)


def A1(j,z,t,U,w0,M):
    """
    Dummy function to extract the A1 coefficient without recalculating the whole continued fraction.
    """
    return An(j,z,t,U,w0,M)[0]


@lru_cache(maxsize=None)
def GMA0(z,t,U,w0,Me,Mh,end,start,p=0):
    """
    Zero phonon Green's function for a Holstein polaron in the IMA0 approximation.
    Based on EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    M = sqrt(Me**2 + Mh**2)
    m = Me + Mh
    dim = len(start)
    centre = (0,)*dim
    SigMA = M*A1(centre,z,t,0.,w0,M)
    sites = neighbor([centre], p)
    vl = diag([M*A1(d,z,t,U,w0,M)-SigMA if d!=centre
                   else m*A1(d,z,t,U,w0,m)-SigMA
                   for d in sites])
    glk = array([[Gu(z-SigMA,t,U,a,b) for b in sites] for a in sites])
    gik = array([Gu(z-SigMA,t,U,end,b) for b in sites])
    Gil = gik.dot(inv(identity(len(sites))-vl.dot(glk)))
    glj = array([[Gu(z-SigMA,t,U,a,start)] for a in sites])
    gij = Gu(z-SigMA,t,U,end,start)
    Gij = gij + Gil.dot(vl).dot(glj)
    return complex(Gij)


@lru_cache(maxsize=None)
def XAS(gam,t,U,w0,Me,Mh=0,p=0,d=2):
    """
    Returns the location of the maximum of the 0 phonon Green's function
    ('XAS experiment') plus the broadening Gamma.
    """
    centre = (0,)*d
    gam = abs(gam)
    res = minimize_scalar(lambda z: imag(GMA0(z+gam*1j,t,U,w0,Me,Mh,centre,centre,p)))
    return res.x + gam*1j


@lru_cache(maxsize=None)
def FMA0(z,q,t,U,w0,Me,Mh=0,p=0):
    """
    All Fn generalized Green's functions for a Holstein polaron in the IMA0 approximation.
    Based on EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    M = sqrt(Me**2 + Mh**2)
    m = Me + Mh
    dim = len(q)
    centre = (0,)*dim
    SigMA = M*A1(centre,z,t,0.,w0,M)
    sites = neighbor([centre], p)
    ann = [An(d,z,t,U,w0,M) if d!=centre
               else An(d,z,t,U,w0,m)
               for d in sites] #for the Fn functions
    nn = min([len(a) for a in ann])
    ann = array([a[:nn] for a in ann])
    vl = diag([M*A1(d,z,t,U,w0,M)-SigMA if d!=centre
               else m*A1(d,z,t,U,w0,m)-SigMA
               for d in sites])
    glk = array([[Gu(z-SigMA,t,U,a,b) for b in sites] for a in sites])
    gik = array([Gu(z-SigMA,t,U,centre,b) for b in sites])
    Gil = gik.dot(inv(identity(len(sites))-vl.dot(glk)))
    gli = array([[exp(-1j*dot(q,a))*Gu(z-n*w0,t,U,a,centre)/Gu(z-n*w0,t,U,a,a)
                      for a in sites] for n in range(1,nn+1)])
    Fnil = array(list(it.accumulate(ann.T,op.mul,initial=Gil)))
    Fnil[1:] = Fnil[1:]*gli #add initial propagator and phase factor
    Fnil[0] = GMA0(z,t,U,w0,Me,Mh,centre,centre,p) #replace G_il with G_ii
    return Fnil


def IMA0(om,z,q,t,U,w0,Me,Mh=0,p=0):
    """RIXS intensity calculated using continued fractions in the IMA0 approximation."""
    d = len(q)
    centre = (0,)*d
    Fn = FMA0(z,q,t,U,w0,Me,Mh,p)
    if om==None: #spectral amplitudes starting from n=0
        In = array([(abs(Fn[0,0])**2)]+[abs(sum([product(g) for g in it.product(conj(f),f)])) for f in Fn[1:]])
        #In = array([(abs(Fn[0,0])**2)]+[sum(abs(f)**2) for f in Fn[1:]]) #for q-integrated
        return In
    else: #RIXS cross-section, without the 0 phonon peak
        In = array([sum([product(g) for g in it.product(conj(f),f)])/(om-n*w0)
                        for n,f in enumerate(Fn) if n>=1])
        return -imag(sum(In))/pi
