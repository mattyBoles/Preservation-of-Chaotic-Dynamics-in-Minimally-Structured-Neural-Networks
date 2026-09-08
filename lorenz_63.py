import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
import random

class lorenz_63():

    '''
    Class to generate trajectories of a Lorenz attrcator, based on Lorenz63, the simplified weather model describing fluid motion and heat:
    
    dx/dt = sigma(y - x)
    dy/dt = x(rho - z) - y
    dz/dt = xy - beta*z

    Here, x is the intensity of the fluid's motion.
    y is the temperature difference between the rising and falling fluid currents.
    z is the distortion of the vertical temperature profile from a stright line.

    '''
    def __init__(self,
                 sigma: float = 10,
                 rho: float = 28,
                 beta: float = 8/3,
                 dt = 0.01):
        
        '''
        Initiates Lorenz generator class for given parameters:

        Inputs:
            sigma (float): Prandtl number, the ratio of a fluid's momentum to its heat diffusion.
            rho (float): Rayleigh number, the temperature difference between top and bottom fluid layer.
            beta (float): The ratio of the width to the height of the convection cell.
        '''
        
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt

        self.dxdt = lambda x, y, z: self.sigma * (y - x)
        self.dydt = lambda x, y, z: (x * (self.rho - z)) - y
        self.dzdt = lambda x, y, z: (x * y) - (self.beta * z)

    def calc_derivatives(self,
            x: np.ndarray) -> np.ndarray:
        ''' 
        Calculates and returns xdot_, the vector time derivative of the vector [x,y,z]

        Inputs:
            x (tuple[float, float, float]): The input vector, [x, y, z]
        
        Returns:
            np.ndarray: The instantanious time derivatives at the input position, in shape [3,]
        '''
        
        x_dot = self.dxdt(x[0], x[1], x[2])
        y_dot = self.dydt(x[0], x[1], x[2])
        z_dot = self.dzdt(x[0], x[1], x[2])

        return np.array([x_dot, y_dot, z_dot])
    
    @staticmethod
    def rk4(f: Callable,
            x: np.ndarray,
            dt: float) -> np.ndarray:
        
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
        k1 = f(x)

        k2 = f(x + (k1 * dt/2))

        k3 = f(x + (k2 * dt/2))

        k4 = f(x + (k3*dt))

        x = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6

        return x
    

    def J(self,
          x_: np.ndarray) -> np.ndarray:
        '''
        Calculates the instantanious Jacobian of the input vector.
        '''
        
        x, y, z = x_[0], x_[1], x_[2]
        return np.array([[-1*self.sigma, self.sigma, 0],
                           [self.rho - z, -1, -1*x],
                           [y, x, -1*self.beta]])

    def rk4_matrix_and_x(self,
                         f: Callable,
                         J: Callable,
                         x: np.ndarray,
                         U: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        
        '''
        Runge-Kutta 4th order time stepping scheme, to be used as ground truth. https://www.geeksforgeeks.org/dsa/runge-kutta-4th-order-method-solve-differential-equation/
        But we also calculate the time stepped M, the tangent propogator. We can then find the instantaneous Singualr Values, which then inform our Lyapunov Spectrum.
        
        Inputs:
            f (Callable): The derivative function of x, in this case the Lorenz63 system, self.calc_derivatives, which retruns dx_/dt, at the input vector.
            J (Callable): The Jacobian function of the system, which returns a matrix with partial derivatives calculated at x.
            x_ (np.ndarray): The current position vector of the system.
            U (np.ndarray): The current tangent propogater of the system.
            dt (float): The delta t, timestep.
        
        Returns:
            tuple[np.ndarray, np.ndarray]:
                x_new: The new position vector after timestepping.
                U_new: The new tangent propogater after timestepping.
        '''

        if U is None:
            U = np.eye(3)

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
    

    def generate_trajectory(self,
                            x0: np.ndarray,
                            n_steps: int,
                            dt: float = 0.01) -> np.ndarray:
        
        '''
        Generates a trajetcory by tiemstepping in rk4.

        Input:
            x0 [np.ndarray]: The starting position vector, np.array([x, y, z])
            n_steps (int): The number of steps to iterate.
            dt (float): The lengfth of timesteps.
        
        Returns:
            np.ndarray: The trajectory, shape [n_steps + 1, 3]
        '''
        
        x_out = np.empty([n_steps+1, 3])
        x_out[0] = x0

        for step_idx in range(1,n_steps+1):
            x_out[step_idx] = self.rk4(self.calc_derivatives, x = x_out[step_idx - 1], dt = dt)
        
        return x_out
    
    @staticmethod
    def plot(png_name: str,
             traj1: np.ndarray,
             traj2: Optional[np.ndarray] = None) -> None:

        '''
        Plots 1 or 2 trajectories of Lorenz.

        Inputs:
            png_name (str): The out file path that the png will be saves as.
            traj1 (np.ndarray): The first trajectory to plot, in shape (n_steps, 3) 
            traj2 (Optional[np.ndarray]): The second trajectory to plot, same shape as traj1.
        
        '''        
        x, y, z = traj1[:,0], traj1[:,1], traj1[:,2]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x, y, z, lw=1, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Lorenz Attractor')

        if traj2 is not None:
            x, y, z = traj2[:,0], traj2[:,1], traj2[:,2]
            ax.plot(x, y, z, lw=1, c='r', alpha=0.5)    
        #plt.show()  
        plt.savefig(png_name)
        plt.close('all')
        


        
    def find_lyapunov_exponent(self,
                               x: np.ndarray,
                               n_transient: int = 5000,
                               n_steps: int = 10000,
                               d0: np.ndarray = None, 
                               re_norm_steps: int = 10, 
                               dt: float = 0.01) -> float:
        '''
        Estimates the Dominant Lyapunov Exponent of the system:
        1. Integrate over transient period and throw away.
        2. Pertubate x0 by a small margin, d0.
        3. Integrate over  small number of steps, t.
        4. Find an estaimet of lambda1, lambda = ln(delta/d0) [/ (t*dt), bu we do this at the end for compute].
        5. Normalise the delta to |d0|, but in the same direction as delta. We do this as we need a small enough pertuabtion to assume linearity, and too many timesteps and it's no lomnger close enough.
        6. Repeat

        Inputs:
            x (np.ndarray): Starting posiiton vector, [x,y,z].
            n_transient (int): Number of timesteps to disregard, to ensure accumulating lambdas are on the attractor.
            n_steps (int): Number of timesteps total to intergrate over.
            d0 (float): Starting pertubation length.
            re_norm_steps: (int): Number of timesteps of each renormalisation cycle.
            dt (float): Timestep length.

        Returns:
            float: Estimate of dominant Lyapunov Exponent.

        '''
        if d0 is None:
            d0 = np.array([1e-8, 0.0, 0.0])
        #Let settle
        for _ in range(n_transient):
            x = self.rk4(f = self.calc_derivatives, x = x, dt = dt)

        lambda_cum = 0
        count = 0

        x_pertubated = x + d0

        d0_abs = np.linalg.norm(d0)

        for i in range(n_steps):
            x = self.rk4(f=self.calc_derivatives, x = x, dt = dt)
            x_pertubated = self.rk4(f = self.calc_derivatives, x = x_pertubated, dt = dt)

            if (i+1) % re_norm_steps == 0:
            
                delta_vec = (x_pertubated - x)
                delta_abs = np.linalg.norm(delta_vec)

                if delta_abs == 0:
                    raise ValueError("Perturbation collapsed to zero")

                lambda_cum += np.log(delta_abs / d0_abs)

                delta_vec *= d0_abs / delta_abs
                x_pertubated = x + delta_vec
                count += 1

        return lambda_cum/(count*dt*re_norm_steps)
    

    def find_lyapunov_spectrum(self,
                               x: np.ndarray,
                               transient_steps: int = 5000,
                               trajectory_steps: int = 10000,
                               QR_steps: int = 10,
                               dt: float = 0.01) -> dict:
        '''
        Method to run find the lyapunov spectrum, and singular values, of lorenz via accumulated SVs and Qr decomp of the Jacobian.
        Jacobian is found every timestep and singular values are added to the list. Q is updated by the Jacobian via rk4 every step and
        is renormalised every so often via QR decomposition. Then, at the end, they are logged and averaged to find Lyapunov spectrum.

        Inputs:
            x (np.ndarray): Starting point, shape (3,) for (x,y,z).
            transient_steps (int): The number of steps to through away at the start to ensure we only start counting when we are on the attractor.
            trajectory_steps (int): The number of timesteps (following transient) to iterate and average over.
            QR_steps (int): The number of steps before Q is renormalised via QR decomp.
            dt (float): The timestep.
        
        Returns:
            dict:
                'lyapunov_spectrum': np.ndarray, shape (3,) containing the found long-term averaged lyapunov spectrum.
                'zdot': The time derivative of z, as an array.
                'singular_values': A np.ndarray of shape (3, trajectory_steps//QR_steps).
                'l_vectors': A list of length (trajectory_steps), containing U, the Left Vectors of SVD, which refer to the output directions fo teh singular values.
                'r_vectors': A list of length (trajectory_steps), containing Vt, the Right Vectors of SVD, which show the input directions of the singular values.
                'x_': A list of length (trajectory_steps), contianing the posiiton vector at each point.
        '''


        #Initialise
        Q = np.identity(3)
        n_qr_samples = trajectory_steps // QR_steps
        lambda_ = np.empty((3, n_qr_samples))
        singular_values, l_vectors, r_vectors, zdot = [], [], [], []
        count = 0
        x_ = []

        for i in range(transient_steps):
            x, Q = self.rk4_matrix_and_x(f = self.calc_derivatives, J = self.J, x = x, U = Q)

            if (i+1) % QR_steps == 0:
                Q, R = np.linalg.qr(Q)
                

        for i in range(trajectory_steps):
            Phi = np.eye(3)
            _, Phi = self.rk4_matrix_and_x(f = self.calc_derivatives, J = self.J, x = x, U = Phi)
            x, Q = self.rk4_matrix_and_x(f = self.calc_derivatives, J = self.J, x = x, U = Q)
            x_.append(x)
            

            U, S, Vt = np.linalg.svd(Phi)

            singular_values.append(S)
            l_vectors.append(U)
            r_vectors.append(Vt)
            zdot.append(self.calc_derivatives(x)[2])

            if (i+1) % QR_steps == 0:
                
                Q, R = np.linalg.qr(Q)
                lambda_[:, count] = np.log(np.abs(np.diag(R))) / (dt * QR_steps)
                count += 1
        
        lambda_ = lambda_[:, :count]
        lyapunov_spectrum = np.mean(lambda_, axis=1)



        return {
            'lyapunov_spectrum':lyapunov_spectrum,
            'zdot':zdot,
            'singular_values':singular_values,
            'l_vectors':l_vectors,
            'r_vectors':r_vectors,
            'x_': x_
        }

    def plot_svs(self,
                 x = np.ndarray,
                 transient_steps: int = 5000,
                 trajectory_steps: int = 1000,
                 ):

        singular_values= []
        for _ in range(transient_steps):
            x, _ = self.rk4_matrix_and_x(f = self.calc_derivatives, J = self.J, x = x)

                

        for _ in range(trajectory_steps):
            Phi = np.eye(3)
            x, Phi = self.rk4_matrix_and_x(f = self.calc_derivatives, J = self.J, x = x, U = Phi)
            
            U, S, Vt = np.linalg.svd(Phi)

            singular_values.append(S)
        svs = np.asarray(singular_values)

        fig, ax = plt.subplots(1,1, figsize=(12,4))

        ax.plot(svs[:1000,0], label = 'SV1')
        ax.plot(svs[:1000,1], label = 'SV2')
        ax.plot(svs[:1000,2], label = 'SV3')


        ax.set_title('Singular Values of True Lorenz')
        ax.set_xlabel(f"Timestep, $\Delta t = 0.01$")
        ax.set_ylabel(f"Singular Value")
        ax.legend()
        plt.show()


            

        
if __name__ == '__main__':
    
    generator = lorenz_63()

    generator.plot_svs(x=np.array([1,1,1]))
    

    




