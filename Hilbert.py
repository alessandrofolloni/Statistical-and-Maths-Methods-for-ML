import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

n=13
A_full=scipy.linalg.hilbert(n)
rel_err=np.zeros((n-2,))
k_2=np.zeros((n-2,))
k_inf=np.zeros((n-2,))

for i in range(4,n):
    x_true=np.ones((i,))
    A=A_full[:i,:i]
    b=x_true@A

    k_2[i-2]=np.linalg.cond(A,2)
    k_inf[i-2]=np.linalg.cond(A,np.inf)
    x=np.linalg.solve(A,b)

    rel_err[i-2]=np.linalg.norm(x_true-x,2)/np.linalg.norm(x_true,2)


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Relative error")
plt.xlabel("n")
plt.ylabel("Relative Error")
plt.plot(np.arange(2,n), rel_err)
plt.grid()

plt.subplot(1, 2, 2)
plt.title("condition numbers")
plt.xlabel("n")
plt.ylabel("k")
plt.plot(np.arange(2,n), k_2)
plt.plot(np.arange(2,n), k_inf)
plt.legend(['k_2', 'k_inf'])
plt.grid()

plt.show()

