import numpy as np
cimport numpy as np
from joblib import Parallel, delayed
from itertools import product

# Type for Cython NumPy acceleration
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef extern from "math.h" nogil:
    double exp(double)

cdef double Exp(double x) nogil:
    return exp(x)

#@cython.boundscheck(False)
cdef class pulse:
    
    cdef int N
    cdef np.ndarray t
    cdef np.ndarray y
    cdef double s_y2, t_min, t_max
    cdef double pi
    
    def __cinit__(self, np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=1] t):

        self.N = len(y)

        self.y = y
        self.t = t
        self.s_y2 = y.dot(y)

        self.t_min = t[3]
        self.t_max = t[-3]
        
        self.pi = np.pi
        
    cdef np.ndarray[DTYPE_t, ndim=1]  __w(self, double delta, double t0):
        cdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros(self.N)
        
        cdef int i
        for i in range(self.N):
            v[i] = exp(-delta*(self.t[i]-t0))
        
        v[np.where(self.t < t0)] = 0    

        return v
        
    cdef np.ndarray[DTYPE_t, ndim=1]  __phi(self, double omega):
        return np.cos(2*self.pi*omega*self.t)
    
    cdef np.ndarray[DTYPE_t, ndim=1]  __psi(self, double omega):
        return np.sin(2*self.pi*omega*self.t)
        
    cdef double llhood(self, double omega, double delta, double t0):

        cdef np.ndarray[DTYPE_t, ndim=1] w = self.__w(delta, t0)
        cdef np.ndarray[DTYPE_t, ndim=1] phi = self.__phi(omega)
        cdef np.ndarray[DTYPE_t, ndim=1] psi = self.__psi(omega)
        
        cdef np.ndarray[DTYPE_t, ndim=1] d = w*phi
        cdef np.ndarray[DTYPE_t, ndim=1] e = w*psi
        
        cdef double sum_d2 = d.dot(d)
        cdef double sum_e2 = e.dot(e)
        cdef double sum_yd = self.y.dot(d)
        cdef double sum_ye = self.y.dot(e)
        
        
        cdef double Z0 = np.log(d.dot(d)) + np.log(e.dot(e))
        
        cdef double ll
        cdef double neginf = -np.infty
        cdef double Z = self.s_y2 - ((sum_yd**2)/sum_d2 + (sum_ye**2)/sum_e2)
        if Z > 0:
            ll = 0.5*Z0 - ((self.N+1)/2)*np.log(Z)
            #ll =  - ((self.N+1)/2)*np.log(Z)
        else:
            #print("Z = {}".format(Z))
            ll = neginf

        return ll
    
    cpdef get_llhood_t(self, double omega, double delta, int npoints = 100):
        
        cdef np.ndarray[DTYPE_t, ndim=1] _t = np.linspace(self.t_min, self.t_max, npoints)
        cdef np.ndarray[DTYPE_t, ndim=1] L = np.zeros(npoints)
        cdef int i
        for i in range(npoints):
            L[i] = self.llhood(omega, delta, _t[i])
            
        return L

    cpdef double _M(self, double omega, double l, int npoints = 100):
        cdef np.ndarray[DTYPE_t, ndim=1] p = self.get_llhood_t(omega, l, npoints)

        cdef double mp = np.max(p)
        cdef double Z = np.log(np.sum(np.exp(p - mp))) + mp

        return Z

    

    cpdef pulsogram(self, double omega_min, double omega_max, int n_omega, double delta_min, double delta_max, int n_delta, int n_grid_t):

        cdef np.ndarray[DTYPE_t, ndim=1] linomega = np.linspace(omega_min, omega_max, n_omega)
        cdef np.ndarray[DTYPE_t, ndim=1] lindelta = np.linspace(delta_min, delta_max, n_delta)
        cdef np.ndarray[DTYPE_t, ndim=2] lM = np.zeros((n_omega, n_delta))

        from cython.parallel cimport prange, parallel
        from cython.parallel import prange, parallel

        cdef int i, j
        #with nogil, parallel():
        #    for i in prange(n_omega):
        #        for j in prange(n_delta):
        #            lM[i,j] = self._M(linomega[i], lindelta[j], n_grid_t)
        for i in range(n_omega):
            for j in range(n_delta):
                lM[i,j] = self._M(linomega[i], lindelta[j], n_grid_t)

        return lM, linomega, lindelta


cdef class abubble:
    
    cdef int N
    cdef np.ndarray t, y
    cdef double s_y2, t_min, t_max, __gamma, __rho, __p    
    
    def __cinit__(self, np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=1] t):

        self.N = len(y)

        self.y = y
        self.t = t
        self.s_y2 = y.dot(y)

        self.t_min = t[3]
        self.t_max = t[-15]
        
        # Specific heat
        self.__gamma = 1.4
        
        # Density
        self.__rho = 998.
        
        # Atmospheric pressure
        self.__p = 101325.   

    cdef np.ndarray[DTYPE_t, ndim=1] __w(self, double delta, double t0):
        cdef np.ndarray[DTYPE_t, ndim=1] v = np.exp(-delta*(self.t-t0))
        v[np.where(self.t < t0)] = 0
        return v
    
    cdef np.ndarray[DTYPE_t, ndim=1] w(self, double delta, double t0):
        return self.__w(delta, t0)
        
    cdef np.ndarray[DTYPE_t, ndim=1] __phi(self, double omega):
        return np.cos(2*np.pi*omega*self.t)
    
    cdef np.ndarray[DTYPE_t, ndim=1] phi(self, double omega):
        return self.__phi(omega)
    
    cdef np.ndarray[DTYPE_t, ndim=1] __psi(self, double omega):
        return np.sin(2*np.pi*omega*self.t)
    
    cdef np.ndarray[DTYPE_t, ndim=1] psi(self, double omega):
        return self.__psi(omega)
        
    cdef np.ndarray[DTYPE_t, ndim=1] __omega(self, double r):
        return (1/(2*np.pi*r))*np.sqrt((3*self.__gamma*self.__p)/self.__rho)
    
    cdef np.ndarray[DTYPE_t, ndim=1] omega(self, double r):
        return self.__omega(r)    
    
    cdef double __delta(self, r):
        cdef np.ndarray[DTYPE_t, ndim=1] f0 = self.__omega(r)
        return np.pi * (0.014 + (1.1 * 10**(-5))*f0) * f0
    
    cdef double delta(self, r):
        return self.__delta(r)

    
    cdef double llhood(self, double r, double t0):

        # Frequency
        cdef np.ndarray[DTYPE_t, ndim=1] omega = self.__omega(r)
        
        # Dumping
        cdef double delta = self.__delta(r)
        
        cdef np.ndarray[DTYPE_t, ndim=1] w = self.__w(delta, t0)
        cdef np.ndarray[DTYPE_t, ndim=1] phi = self.__phi(omega)
        cdef np.ndarray[DTYPE_t, ndim=1] psi = self.__psi(omega)
        
        cdef np.ndarray[DTYPE_t, ndim=1] d = w*phi
        cdef np.ndarray[DTYPE_t, ndim=1] e = w*psi
        
        cdef double sum_d2 = d.dot(d)
        cdef double sum_e2 = e.dot(e)
        cdef double sum_yd = self.y.dot(d)
        cdef double sum_ye = self.y.dot(e)
        
        cdef double neginf = -np.infty
        
        cdef double Z0 = np.log(d.dot(d)) + np.log(e.dot(e))
        
        cdef double Z = self.s_y2 - ((sum_yd**2)/sum_d2 + (sum_ye**2)/sum_e2)
        cdef double ll
        if Z > 0:
            ll = -0.5 * Z0 - ((self.N+1)/2)*np.log(Z)
            #ll =  - ((self.N+1)/2)*np.log(Z)
        else:
            print("Z = {}".format(Z))
            ll = neginf

        return ll
    
    cdef get_llhood_t(self, double r, int npoints = 100):
        
        cdef np.ndarray[DTYPE_t, ndim=1] _t = np.linspace(self.t_min, self.t_max, npoints)
        
        cdef np.ndarray[DTYPE_t, ndim=1] L = np.zeros(npoints)
        cdef int i
        
        from cython.parallel cimport prange, parallel
        from cython.parallel import prange, parallel        
        
        for i in range(npoints):
            L[i] = self.llhood(r, _t[i])
            
        return _t, L

    cdef np.ndarray[DTYPE_t, ndim=1] _M(self, r, n_grid_t):
        cdef np.ndarray[DTYPE_t, ndim=1] p
        _, p = self.get_llhood_t(r, n_grid_t)

        return p

    

    cdef np.ndarray[DTYPE_t, ndim=2] bubblegram(self, double r_min, double r_max, int n_r, int n_grid_t = 100):

        cdef np.ndarray[DTYPE_t, ndim=1] linr = np.linspace(r_min, r_max, n_r)

        from joblib import Parallel, delayed

        #lM = Parallel(n_jobs=-1)(delayed(self._M)(ol, n_grid_t) for ol in linr)
        #cdef np.ndarray[DTYPE_t, ndim=2] M = np.array(lM)
        #M = np.reshape(M, [n_r, n_grid_t])    
        cdef int i, j
        cdef np.ndarray[DTYPE_t, ndim=2] M = np.zeros((n_r, n_grid_t))
        for i in range(n_r):
                M[i,:] = self._M(linr[i], n_grid_t)
                

        return M