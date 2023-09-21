"""
Lang-Firsov theory and RIXS cross-section for a Holstein phonon model.
"""

from numpy import *
from scipy.special import *

from builtins import sum #for bringing back functions shadowed by numpy
from functools import *

from propagators import Gu


@lru_cache(maxsize=None)
def Breal(m,n,alpha):
    '''Simplified symmetrized FC factor for real alpha.'''
    if m < n:
        return Breal(n,m,alpha)
    L = eval_genlaguerre(n,m-n,abs(alpha)**2)
    E = exp((m-n)*log(alpha)-abs(alpha)**2/2-0.5*(gammaln(m+1)-gammaln(n+1)))
    return (-1)**m*L*E


@lru_cache(maxsize=None)
def bgb(z,t,U,w0,alpha,nf,n0,nt,end,start,nph=None):
    g = alpha**2
    if nph==None:
        nph = 10*int(ceil(g))
    return sum([Breal(nf,n,alpha)
                    *Gu(z-w0*(nt-n0+n-g),t,U,end,start)
                    *Breal(n,n0,alpha) for n in range(nph+1)])


def F0(om,z,t,U,w0,alpha,N,nph=None):
    m = arange(N+1)
    G = array([bgb(z,t,U,w0,alpha,n,0,0,(0,0),(0,0),nph) for n in m])
    F = abs(G)**2
    if om==None: #spectral amplitudes
        return F
    else: #RIXS cross-section, without the 0 phonon peak
        I = array([f/(om-n*w0) for n,f in zip(m,F) if n>0])
        return -imag(sum(I))/pi
