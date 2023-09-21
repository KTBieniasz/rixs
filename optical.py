"""
Functions for calculating the RIXS cross-section, XAS and related quantitites for an optical dispersive phonon model.
"""

from numpy import *
from scipy.linalg import inv
from scipy.special import *
from scipy.optimize import *

from builtins import sum, round #for bringing back functions shadowed by numpy
import itertools as it
from functools import *
from collections import Counter

from propagators import Gu, neighbor, Eps


def Eph(k,w0,w1):
    """Optical phonon dispersion."""
    return w0+Eps(k,w1)


@lru_cache(maxsize=None)
def energies(eps,args,n,N,q=2,f=3):
    """
    Generates energy and multiplicity arrays to optimize n-dim integration over d-dim 2N-point BZ, with (small) energy shifts caused by optical phonon dispersion. 
    Optional q-dependence is provided for (n-1)-dim integration with constraint q=\sum_l k_l.
    """
    ### For q-indep
    if type(q) is int:
        d = q
        if n==0:
            E = Counter({0.:1})
        if n==1:
            K = Counter(abs(linspace(-pi,pi,4*N,endpoint=None)))
            Q = Counter({k:factorial(d)*product([K[q] for q in k])
                         /product(factorial(list(Counter(k).values())))
                         for k in it.combinations_with_replacement(K,d)})
            E = Counter()
            for k,m in Q.items():
                en = round(eps(k,*args),f)
                E[en] += m/(4*N)**d
        if n>1:
            En = energies(eps,args,n-1,N,d,f=f)
            Er = energies(eps,args,1,N,d,f=f)
            E = Counter()
            for en,er in it.product(En,Er):
                E[round(en+er,f)] += En[en]*Er[er]
    ### For q-dep
    else:
        q = array(q)
        d = len(q) #override d to dim of q
        n -= 1 #reduce by one due to delta_qk
        K = linspace(-pi,pi,4*N,endpoint=None) #1-dim BZ
        Q = it.product(K,repeat=d) #d-dim K
        E = Counter()
        if n>0:
            for qq in it.product(Q,repeat=n):
                qq = vstack([(q-(array(qq)+pi).sum(axis=0))%(2*pi)-pi, qq])
                en = round(sum([eps(k,*args) for k in qq]), f)
                E[en] += 1/(4*N)**(d*n)
        else:
            E[eps(q,*args)] = 1.
    return E


def Gnbar(n,w0,w1,z,t,U,a,b,N=2):
    """
    Performs n-dimensional trapezoidal integration of the Gu inhomogenous Green's function using the k-points multiplicities.
    """
    d = len(a)
    E = energies(Eph,(w0,w1),n,N,q=d)
    return sum([m*Gu(z-e,t,U,a,b) for e,m in E.items()])


def In(n,om,z,q,t,U,w0,w1,a,b,N=2):
    """
    The free coefficient corresponding to the n-phonon mode free Green's function integrated using trapezoidal rule and k-points multiplicities.
    """
    d = len(q)
    centre = (0,)*d
    if n==0:
        if all(q==0): #non-zero only when no q-transfer
            if om==None:
                return conj(Gu(z,t,U,a,centre))*Gu(z,t,U,centre,b)
            else:
                return conj(Gu(z,t,U,a,centre))*Gu(z,t,U,centre,b)/om
        else:
            return 0
    else:
        E = energies(Eph,(w0,w1),n,N,q)
        if om==None: #Integrated amplitudes
            return sum([m*conj(Gu(z-e,t,U,a,centre))*Gu(z-e,t,U,centre,b) for e,m in E.items()])
        else: #Integrated lines
            return sum([m*conj(Gu(z-e,t,U,a,centre))*Gu(z-e,t,U,centre,b)/(om-e) for e,m in E.items()])


@lru_cache(maxsize=None)
def Anq(j,z,t,U,w0,w1,M,N=2):
    """
    Continued fraction all coefficients for all generalized Green's functions in the IMA0 approximation for the q dependent case.
    Based on generalization of EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    nph = 4*max(int(ceil((M/w0)**2)), 1)
    A = array([(n, M*Gnbar(n,w0,w1,z,t,U,j,j,N)) for n in range(nph,0,-1)])
    A0 = reduce(lambda x,y: product(y)/(1-y[1]*product(x)), A[1:])
    An = list(it.accumulate(A,
                            lambda x,y: product(y)/(1-y[1]*x),
                            initial=0.))
    A1 = An[-1]
    while abs((A1-A0)/A1)>1e-16:
        nph *= 2
        A = array([(n, M*Gnbar(n,w0,w1,z,t,U,j,j,N)) for n in range(nph,0,-1)])
        An = list(it.accumulate(A,
                                lambda x,y: product(y)/(1-y[1]*x),
                                initial=0.))
        A0 = A1
        A1 = An[-1]
    return array(An[:0:-1])/sqrt(A[::-1,0]) #For Fn normalized by sqrt(n!)


def A1q(j,z,t,U,w0,w1,M,N=2):
    """
    Dummy function to extract the A1 coefficient without recalculating the whole continued fraction.
    """
    return Anq(j,z,t,U,w0,w1,M,N)[0]


@lru_cache(maxsize=None)
def GMAq(z,t,U,w0,w1,Me,Mh,end,start,p=0):
    """
    Zero phonon Green's function for dispersive phonons in the IMA0 approximation.
    Based on EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    M = sqrt(Me**2 + Mh**2)
    m = Me + Mh
    dim = len(start)
    centre = (0,)*dim
    SigMA = M*A1q(centre,z,t,0.,w0,w1,M)
    sites = neighbor([centre], p)
    vl = diag([M*A1q(d,z,t,U,w0,w1,M)-SigMA if d!=centre
                   else m*A1q(d,z,t,U,w0,w1,m)-SigMA
                   for d in sites])
    glk = array([[Gu(z-SigMA,t,U,a,b) for b in sites] for a in sites])
    gik = array([Gu(z-SigMA,t,U,end,b) for b in sites])
    Gil = gik.dot(inv(identity(len(sites))-vl.dot(glk)))
    glj = array([[Gu(z-SigMA,t,U,a,start)] for a in sites])
    gij = Gu(z-SigMA,t,U,end,start)
    Gij = gij + Gil.dot(vl).dot(glj)
    return complex(Gij)


@lru_cache(maxsize=None)
def XASq(gam,t,U,w0,w1,Me,Mh=0,p=0,d=2):
    """
    Returns the location of the maximum of the 0 phonon Green's function
    ('XAS experiment'), plus the broadening Gamma.
    """
    centre = (0,)*d
    gam = abs(gam)
    res = minimize_scalar(lambda z: imag(GMAq(z+gam*1j,t,U,w0,w1,Me,Mh,centre,centre,p)))
    return res.x + gam*1j


@lru_cache(maxsize=None)
def IMAq(N,om,z,q,t,U,w0,w1,Me,Mh=0,p=0):
    """
    Zero phonon Green's function for a Holstein polaron in the IMA0 approximation.
    Based on EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    M = sqrt(Me**2 + Mh**2)
    m = Me + Mh
    dim = len(q)
    centre = (0,)*dim
    sites = neighbor([centre], p)
    SigMA = M*A1q(centre,z,t,0.,w0,w1,M)
    anq = [Anq(d,z,t,U,w0,w1,M) if d!=centre
               else Anq(d,z,t,U,w0,w1,m)
               for d in sites]
    nn = min([len(a) for a in anq])
    anq = array([a[:nn] for a in anq])
    manq = multiply.accumulate(anq, axis=1)
    vl = diag([M*A1q(d,z,t,U,w0,w1,M)-SigMA if d!=centre
               else m*A1q(d,z,t,U,w0,w1,m)-SigMA
               for d in sites])
    glk = array([[Gu(z-SigMA,t,U,a,b) for b in sites] for a in sites])
    Tlk = inv(identity(len(sites))-vl @ glk)
    glj = array([[Gu(z-SigMA,t,U,a,centre)] for a in sites])
    Dlj = (identity(len(sites)) + glk @ Tlk @ vl) @ glj
    if om==None:
        Ikl = array([[[In(n,om,z,q,t,U,w0,w1,a,b)*conj(manq[i,n-1]*exp(-1j*(dot(q,a)))/Gnbar(n,w0,w1,z,t,U,a,a))*manq[j,n-1]*exp(-1j*(dot(q,b)))/Gnbar(n,w0,w1,z,t,U,b,b) for j,b in enumerate(sites)] for i,a in enumerate(sites)] for n in range(1,N+1)])
        Gij = array([complex(conj(Dlj.T) @ I @ Dlj) for I in Ikl])
        return real(Gij)
    else:
        Ikl = array([[sum([In(n,om,z,q,t,U,w0,w1,a,b)*conj(manq[i,n-1]*exp(-1j*(dot(q,a)))/Gnbar(n,w0,w1,z,t,U,a,a))*manq[j,n-1]*exp(-1j*(dot(q,b)))/Gnbar(n,w0,w1,z,t,U,b,b) for n in range(1,N+1)]) for j,b in enumerate(sites)] for i,a in enumerate(sites)])
        Gij = conj(Dlj.T) @ Ikl @ Dlj
        return -imag(complex(Gij))/pi
