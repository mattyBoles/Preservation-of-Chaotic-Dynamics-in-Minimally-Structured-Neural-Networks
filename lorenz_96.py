import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import json
from typing import Callable


class lorenz_96():
    def __init__(self, N, F, dt):
        self.N = N
        self.F = F
        self.dt = dt


    def derive(self, x: np.ndarray):

        return (
            (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1)
            - x
            + self.F
        )

    def J(self, x):
        J = np.zeros((self.N, self.N))

        for i in range(self.N):
                J[i, i] = -1
                J[i, (i + 1) % self.N] = x[(i - 1) % self.N]
                J[i, (i - 2) % self.N] = -x[(i - 1) % self.N]
                J[i, (i - 1) % self.N] = x[(i + 1) % self.N] - x[(i - 2) % self.N]
        return J


    def rk4(self,
            f: Callable,
            J: Callable,
            x: np.ndarray,
            U: np.ndarray = None) -> np.ndarray:
    
        '''
        Runge-Kutta 4th order time stepping scheme, to be used as ground truth.
        https://www.geeksforgeeks.org/dsa/runge-kutta-4th-order-method-solve-differential-equation/

        Inputs: 
            f (Callable): The derivative function of the system.
            x (np.ndarray): The innput posotin vector, shape [x, y, z]
            dt (float): The timestep length.
        
        Returns:
            np.ndarray: The timestepped position vector.
        '''
        if U is None:
            U = np.eye(self.N)

        k1_x = f(x)
        k1_U = J(x) @ U

        k2_x = f(x + (k1_x * self.dt/2))
        k2_U = J(x + (k1_x * self.dt/2)) @ (U + (k1_U * self.dt/2))

        k3_x = f(x + (k2_x * self.dt/2))
        k3_U = J(x + (k2_x * self.dt/2)) @ (U + (k2_U * self.dt/2))

        k4_x = f(x + (k3_x * self.dt))
        k4_U = J(x + (k3_x * self.dt)) @ (U + (k3_U * self.dt))

        x_new = x + (self.dt/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x))
        U_new = U + (self.dt/6 * (k1_U + 2*k2_U + 2*k3_U + k4_U))


        return x_new, U_new


    def find_lyapunov_spectrum(self,
                                   x: np.ndarray,
                                   transient_steps: int = 5000,
                                   trajectory_steps: int = 100000,
                                   QR_steps: int = 10):
        Q = np.eye(self.N)
        n_qr_samples = trajectory_steps//QR_steps
        lambda_ = np.empty((self.N,n_qr_samples))
        count = 0

        for i in range(transient_steps):
            x, Q = self.rk4(f = self.derive, J = self.J, x = x, U = Q)

            if (i+1) % QR_steps == 0:
                            Q, R = np.linalg.qr(Q)

        Q = np.eye(self.N)
        for i in range(trajectory_steps):
            x, Q = self.rk4(f = self.derive, J = self.J, x = x, U = Q)

            if (i+1) % QR_steps == 0:
                            
                Q, R = np.linalg.qr(Q)
                lambda_[:, count] = np.log(np.abs(np.diag(R))) / (self.dt * QR_steps)
                count += 1

        lambda_ = lambda_[:, :count]
        lyapunov_spectrum = np.mean(lambda_, axis=1)
        lyapunov_spectrum = np.sort(lyapunov_spectrum)[::-1]

        return lyapunov_spectrum

    def find_svs(self,
                 x: np.ndarray,
                 transient_steps: int = 5000,
                 trajectory_steps: int = 1000):

        svs = []

        for _ in range(transient_steps):
            x, _ = self.rk4(self.derive, self.J, x)

        for _ in range(trajectory_steps):
            Phi = np.eye(self.N)
            x, Phi = self.rk4(self.derive, self.J, x, Phi)

            S = np.linalg.svd(Phi, compute_uv=False)
            svs.append(S)

        svs = np.asarray(svs)
        return svs

if __name__ == "__main__":
    c = lorenz_96(8,8, 0.01)

    x0 = np.array([8.01, 8, 8, 8, 8, 8, 8, 8], dtype=float)

    lyapunov = c.find_lyapunov_spectrum(x0)
    print(f"Lambda1: {lyapunov[0]}")
    print(f"Lambda2: {lyapunov[1]}")
    print(f"Lambda3: {lyapunov[2]}")
    print(f"Lambda4: {lyapunov[3]}")
    print(f"Lambda5: {lyapunov[4]}")
    print(f"Lambda6: {lyapunov[5]}")
    print(f"Lambda7: {lyapunov[6]}")
    print(f"Lambda8: {lyapunov[7]}")

    svs = c.find_svs(x0)

    fig, ax = plt.subplots()
    ax.plot(svs[:,4], label = 'Lambda5')
    ax.plot(svs[:,5], label = 'Lambda6')
    ax.plot(svs[:,6], label = 'Lambda7')
    ax.plot(svs[:,7], label = 'Lambda8')
    ax.set_xlabel('dt=0.01')
    ax.set_ylabel('SV')
    ax.set_title('SVs of Double Pendulum')
    ax.legend()
    plt.show()

    


