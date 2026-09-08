import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import json
from typing import Callable


class lorenz_96():

    '''
    Class to calculate Lorenz96 trajectories, via RK4, plus related funcs like claucilting Lyapunov Spctrum.
    '''
    def __init__(self,
                 N: int,
                 F: int,
                 dt: float= 0.01) -> None:

        '''
        Inputs:
            N (int): Number of dimenions
            F (int): Constant in the equations
            dt (float): Timestep length 
        '''
        self.N = N
        self.F = F
        self.dt = dt


    def derive(self,
               x: np.ndarray):

        return (
            (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1)
            - x
            + self.F
        )

    def J(self,
          x: np.ndarray) -> np.ndarray:
        '''
        Jacobian
        '''
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
                                QR_steps: int = 10) -> np.ndarray:
        '''
        Finds the Lyapunov Spectrum of L96, via the Benettin method.

        Inputs:
            x (np.ndarray): The starting coordinate, shape (self.N)
            transient_steps (int): The number of timesteps to throw away at the start, to ensure we are on the attractor
            trajectory_steps (int): The number of timesteps (excluding transient) to average teh Lyapunov over.
            QR_steps (int): How often to accumulate diag(R) and re-orthonormalise Q.
        
        Returns:
            np.ndarray: The Lyapunov spectrum of L96, shape (self.N,)
        '''

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
                 trajectory_steps: int = 1000)-> np.ndarray:
        '''
        Finds the singular values of the flow map of L96.

        Inputs:
            x (np.ndarray): The starting coordinate, shape (self.N)
            transient_steps (int): The number of timesteps to throw away at the start, to ensure we are on the attractor
            trajectory_steps (int): The number of timesteps (excluding transient) to find SVs of.
        
        Returns:
            np.ndarray: The singular values of the flow map, of shape (trajectory_steps, self.N)
        '''

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

    def plot(self,
                x: np.ndarray,
                transient_steps: int = 5000,
                trajectory_steps: int = 1000):
        '''
        Plots a heatmap-like plot showing X_i over 1000 timesteps
        '''

        traj = []
        traj.append(x0)

        for _ in range(transient_steps):
            x, _ = self.rk4(self.derive, self.J, x)
        for _ in range(trajectory_steps):
            x, _ = self.rk4(self.derive, self.J, x)
            traj.append(x)

        traj = np.asarray(traj)

        fig, ax = plt.subplots(figsize=(10, 3))

        im = ax.imshow(
            traj[:1000].T,
            aspect='auto',
            origin='lower',
            cmap='RdBu_r',
            interpolation='nearest'
        )

        ax.set_xlabel('Timestep, $\Delta t = 0.01$')
        ax.set_ylabel('Variable $i$')
        ax.set_yticks(range(8))
        ax.set_yticklabels(range(1, 9))
        ax.set_title("Lorenz-96 Values for N = F = 8")

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('$x_i$')

        plt.tight_layout()
        plt.show()


         

if __name__ == "__main__":
    c = lorenz_96(8,8, 0.01)

    x0 = np.array([8.01, 8, 8, 8, 8, 8, 8, 8], dtype=float)

    c.plot(x=x0)
    


    


