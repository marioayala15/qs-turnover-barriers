
import numpy as np
from scipy.optimize import minimize
from mamlib import action, xs, ws, Uon
r=0.5
A=np.array([xs,1.8083669]); B=np.array([xs,ws])
for T,M in [(20,400),(40,800),(80,1600)]:
  for seed in [(0,0),(0.1,0),(0.3,0.2),(-0.02,0)]:
    t=np.linspace(0,1,M+1)[1:-1,None]
    z0=(A+t*(B-A)+np.array(seed)*np.sin(np.pi*t)).ravel()
    res=minimize(action,z0,args=(r,A,B,M,T),jac=True,method="L-BFGS-B",bounds=[(1e-6,None)]*z0.size,options=dict(maxiter=200000,maxfun=400000,ftol=1e-15,gtol=1e-10))
    print(T,M,seed,"V(cand->saddle)=%.6f"%res.fun, res.message)
