import numpy as np
cimport numpy as np
from joblib import Parallel, delayed
from itertools import product

# Type for Cython NumPy acceleration
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class pulse:
    
    cdef int N
    cdef np.ndarray[DTYPE_t, ndim=1] t
    cdef np.ndarray[DTYPE_t, ndim=1] y
    cdef double s_y2, t_min, t_max
    
    def __cinit__(self, np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=1] t):

        self.N = len(y)

        self.y = y
        self.t = t
        self.s_y2 = y.dot(y)

        self.t_min = t[3]
        self.t_max = t[-3]

    cdef np.ndarray[DTYPE_t, ndim=1]  __w(self, double delta, double t0):
        cdef np.ndarray[DTYPE_t, ndim=1] v = np.exp(-delta*(self.t-t0))
        v[np.where(self.t < t0)] = 0
        return v
        
    cdef np.ndarray[DTYPE_t, ndim=1]  __phi(self, double omega):
        return np.cos(2*np.pi*omega*self.t)
    
    cdef np.ndarray[DTYPE_t, ndim=1]  __psi(self, double omega):
        return np.sin(2*np.pi*omega*self.t)
        
    cdef double llhood(self, double omega, double delta, double t0):

        w = self.__w(delta, t0)
        phi = self.__phi(omega)
        psi = self.__psi(omega)
        
        d = w*phi
        e = w*psi
        
        sum_d2 = d.dot(d)
        sum_e2 = e.dot(e)
        sum_yd = self.y.dot(d)
        sum_ye = self.y.dot(e)
        
        
        Z0 = np.log(d.dot(d)) + np.log(e.dot(e))
        
        Z = self.s_y2 - ((sum_yd**2)/sum_d2 + (sum_ye**2)/sum_e2)
        if Z > 0:
            ll = 0.5*Z0 - ((self.N+1)/2)*np.log(Z)
            #ll =  - ((self.N+1)/2)*np.log(Z)
        else:
            print("Z = {}".format(Z))
            ll = -np.infty

        return ll
    
    cpdef get_llhood_t(self, double omega, double delta, int npoints = 100):
        
        cdef np.ndarray[DTYPE_t, ndim=1] _t = np.linspace(self.t_min, self.t_max, npoints)
        cdef np.ndarray[DTYPE_t, ndim=1] L = np.zeros(npoints)
        cdef int i
        for i in range(npoints):
            L[i] = self.llhood(omega, delta, _t[i])
            
        return _t, L

    cpdef double _M(self, double omega, double l, np.ndarray[DTYPE_t, ndim=1] n_grid_t):
        _, p = self.get_llhood_t(omega, l, n_grid_t)

        cdef double mp = np.max(p)
        cdef double Z = np.log(np.sum(np.exp(p - mp))) + mp

        return Z

    

cpdef pulsogram(obj, double omega_min, double omega_max, int n_omega, double delta_min, double delta_max, int n_delta, int n_grid_t = 100):

    cdef np.ndarray[DTYPE_t, ndim=1] linomega = np.linspace(omega_min, omega_max, n_omega)
    cdef np.ndarray[DTYPE_t, ndim=1] lindelta = np.linspace(delta_min, delta_max, n_delta)
    cdef np.ndarray[DTYPE_t, ndim=2] lM = np.zeros(n_omega, n_delta)

    from cython.parallel cimport prange, parallel
    from cython.parallel import prange, parallel

    with nogil, parallel():
        for i in prange(n_omega):
            for j in prange(n_delta):
                lM[i,j] = obj._M(linomega[i], lindelta[j], n_grid_t)

    return lM, linomega, lindelta


class abubble:
    
    def __init__(self, y, t):

        self.N = len(y)

        self.y = y
        self.t = t
        self.s_y2 = y.dot(y)

        self.t_min = t[3]
        self.t_max = t[-15]
        
        # Specific heat
        self.__gamma = 1.4
        
        # Density
        self.__rho = 998
        
        # Atmospheric pressure
        self.__p = 101325        

    def __w(self, delta, t0):
        v = np.exp(-delta*(self.t-t0))
        v[np.where(self.t < t0)] = 0
        return v
    
    def w(self, delta, t0):
        return self.__w(delta, t0)
        
    def __phi(self, omega):
        return np.cos(2*np.pi*omega*self.t)
    
    def phi(self, omega):
        return self.__phi(omega)
    
    def __psi(self, omega):
        return np.sin(2*np.pi*omega*self.t)
    
    def psi(self, omega):
        return self.__psi(omega)
        
    def __omega(self, r):
        return (1/(2*np.pi*r))*np.sqrt((3*self.__gamma*self.__p)/self.__rho)
    
    def omega(self, r):
        return self.__omega(r)    
    
    def __delta(self, r):
        f0 = self.__omega(r)
        return np.pi * (0.014 + (1.1 * 10**(-5))*f0) * f0
    
    def delta(self, r):
        return self.__delta(r)

    
    def llhood(self, r, t0):

        # Frequency
        omega = self.__omega(r)
        
        # Dumping
        delta = self.__delta(r)
        
        w = self.__w(delta, t0)
        phi = self.__phi(omega)
        psi = self.__psi(omega)
        
        d = w*phi
        e = w*psi
        
        sum_d2 = d.dot(d)
        sum_e2 = e.dot(e)
        sum_yd = self.y.dot(d)
        sum_ye = self.y.dot(e)
        
        
        Z0 = np.log(d.dot(d)) + np.log(e.dot(e))
        
        Z = self.s_y2 - ((sum_yd**2)/sum_d2 + (sum_ye**2)/sum_e2)
        if Z > 0:
            ll = -0.5 * Z0 - ((self.N+1)/2)*np.log(Z)
            #ll =  - ((self.N+1)/2)*np.log(Z)
        else:
            print("Z = {}".format(Z))
            ll = -np.infty

        return ll
    
    def get_llhood_t(self, r, npoints = 100):
        
        _t = np.linspace(self.t_min, self.t_max, npoints)
        
        L = np.zeros(npoints)
        for i in range(npoints):
            L[i] = self.llhood(r, _t[i])
            
        return _t, L

    def _M(self, r, n_grid_t):
        _, p = self.get_llhood_t(r, n_grid_t)

        return p

    

def bubblegram(obj, r_min, r_max, n_r, n_grid_t = 100):

    linr = np.linspace(r_min, r_max, n_r)

    from joblib import Parallel, delayed
    from itertools import product

    lM = Parallel(n_jobs=-1)(delayed(obj._M)(ol, n_grid_t) for ol in linr)
    M = np.array(lM)
    M = np.reshape(M, [n_r, n_grid_t])    

    return M, linr