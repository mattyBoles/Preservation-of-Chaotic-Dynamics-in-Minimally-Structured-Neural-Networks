import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import json
from typing import Callable

class Double_Pendulum():

    def __init__(self,
                 M1,
                 M2, 
                 L1,
                 L2,
                 G=9.8,
                 dt=0.01):

        self.m1 = M1
        self.m2 = M2
        self.l1 = L1
        self.l2 = L2
        self.g = G
        self.dt = dt

    
    def th1_dot(self, x):
        return x[1]

    def th2_dot(self, x):
        return x[3]

    def w1_dot(self, x, compute_num_denom = False):
        theta1, w1, theta2, w2 = x[0], x[1], x[2], x[3]

        w1_dot_num =  ((-1*self.g)* (2*self.m1 + self.m2) * np.sin(theta1) 
                - self.m2 * self.g * np.sin(theta1 - 2*theta2)
                - 2*np.sin(theta1 - theta2) * self.m2 * ((w2**2)*self.l2 + (w1**2)*self.l1*np.cos(theta1 - theta2)))

        w1_dot_denom = (self.l1*(2*self.m1 + self.m2 - (self.m2*np.cos(2*theta1 - 2*theta2))))

        if compute_num_denom == False:
            return w1_dot_num / w1_dot_denom
        else:
            return w1_dot_num, w1_dot_denom

    def w2_dot(self, x, compute_num_denom = False):
        theta1, w1, theta2, w2 = x[0], x[1], x[2], x[3]

        w2_dot_num = (2*np.sin(theta1-theta2)
                    *((w1**2)*self.l1*(self.m1+self.m2) + self.g*(self.m1 + self.m2)* np.cos(theta1)
                    + (w2**2) * self.l2 * self.m2 * np.cos(theta1 - theta2)))

        w2_dot_denom = (self.l2*(2*self.m1 + self.m2 - self.m2*np.cos(2*theta1 - 2*theta2)))

        if compute_num_denom == False:
            return w2_dot_num / w2_dot_denom
        else:
            return w2_dot_num, w2_dot_denom

    def dw1_dot_dt1(self, x):
        t1, w1, t2, w2 = x[0], x[1], x[2], x[3]
        f_x, g_x = self.w1_dot(x, compute_num_denom=True)
        d_f_x = (-1*self.g*(2*self.m1 + self.m2)*np.cos(t1) - ((self.m2*self.g*np.cos(t1-2*t2)))
                - (2*np.cos(t1-t2)*self.m2*((w2*w2*self.l2)+w1*w1*self.l1*np.cos(t1-t2)))
                + (2*np.sin(t1-t2)*(w1*w1*self.l1*np.sin(t1-t2)))
                )
        d_g_x = 2*self.l1*self.m2*np.sin(2*t1 - 2*t2)

        return ((g_x * d_f_x) - (f_x * d_g_x))/(g_x * g_x)

    def dw1_dot_dw1(self, x):
        t1, w1, t2, w2 = x[0], x[1], x[2], x[3]
        return (-4*np.sin(t1-t2)*self.m2*w1*self.l1*np.cos(t1-t2))/(self.l1*(2*self.m1 + self.m2 - self.m2*np.cos(2*t1 - 2*t2)))

    def dw1_dot_dt2(self, x):
        t1, w1, t2, w2 = x[0], x[1], x[2], x[3]
        f_x, g_x = self.w1_dot(x, compute_num_denom=True)
        d_f_x = ((2*self.m2*self.g*np.cos(t1-2*t2))
                + 2* np.cos(t1-t2) * self.m2 *( w2*w2*self.l2 + w1*w1*self.l1*np.cos(t1-t2))
                - 2*np.sin(t1-t2) * self.m2* (w1*w1*self.l1*np.sin(t1-t2))
                )
        d_g_x = -2*self.m2*self.l1*np.sin(2*t1-2*t2)
        return ((g_x * d_f_x) - (f_x * d_g_x))/(g_x * g_x)

    def dw1_dot_dw2(self,x):
        t1, w1, t2, w2 = x[0], x[1], x[2], x[3]
        return (-4*np.sin(t1-t2)*self.m2*w2*self.l2)/(self.l1*(2*self.m1 + self.m2 - (self.m2*np.cos(2*t1 - 2*t2))))


    def dw2_dot_dt1(self, x):
        t1, w1, t2, w2 = x[0], x[1], x[2], x[3]
        f_x, g_x = self.w2_dot(x, compute_num_denom=True)
        d_f_x = (
            2*np.cos(t1-t2) * (w1*w1*self.l1*(self.m1+self.m2)
                                + self.g * (self.m1 + self.m2)* np.cos(t1)
                                + w2*w2*self.l2*self.m2*np.cos(t1-t2))
            - 2*np.sin(t1-t2)* (self.g * (self.m1 + self.m2) * np.sin(t1)
                                + w2*w2*self.l2*self.m2*np.sin(t1-t2))
                )
        d_g_x = 2*self.l2*self.m2*np.sin(2*t1 - 2*t2)
        return ((g_x * d_f_x) - (f_x * d_g_x))/(g_x * g_x)

    def dw2_dot_dw1(self, x):
        t1, w1, t2, w2 = x[0], x[1], x[2], x[3]
        return ((4 * np.sin(t1-t2) * w1 * self.l1 * (self.m1 + self.m2)) 
                / (self.l2 * (2*self.m1 + self.m2 - self.m2 * np.cos(2*t1 - 2*t2)))
                )

    def dw2_dot_dt2(self, x):
        t1, w1, t2, w2 = x[0], x[1], x[2], x[3]
        f_x, g_x = self.w2_dot(x, compute_num_denom=True)
        d_f_x = (
                -2*np.cos(t1-t2) * (w1*w1*self.l1*(self.m1 + self.m2)
                                    + self.g * (self.m1 + self.m2) * np.cos(t1)
                                    + w2*w2*self.l2*self.m2*np.cos(t1-t2))
                + 2*np.sin(t1-t2) * (w2*w2*self.l2*self.m2*np.sin(t1-t2))
        )

        d_g_x = -2*self.l2*self.m2*np.sin(2*t1 - 2*t2)
        return ((g_x * d_f_x) - (f_x * d_g_x))/(g_x * g_x)

    def dw2_dot_dw2(self, x):
        t1, w1, t2, w2 = x[0], x[1], x[2], x[3]
        return ((4* np.sin(t1-t2) * w2 * self.l2 * self.m2 * np.cos(t1-t2))
                / (self.l2 * (2*self.m1 + self.m2 - self.m2*np.cos(2*t1 - 2*t2)))
        )

    def J(self,x):
        return np.array([[0,1,0,0],
                    [self.dw1_dot_dt1(x), self.dw1_dot_dw1(x), self.dw1_dot_dt2(x), self.dw1_dot_dw2(x)],
                    [0,0,0,1],
                    [self.dw2_dot_dt1(x), self.dw2_dot_dw1(x), self.dw2_dot_dt2(x), self.dw2_dot_dw2(x)]])


    def deriv(self,x):
        return np.array([self.th1_dot(x), self.w1_dot(x), self.th2_dot(x), self.w2_dot(x)])


    def rk4(self,
            f: Callable,
            J: Callable,
            x: np.ndarray,
            U: np.ndarray = np.eye(4)) -> np.ndarray:
    
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
        Q = np.eye(4)
        n_qr_samples = trajectory_steps//QR_steps
        lambda_ = np.empty((4,n_qr_samples))
        count = 0

        for i in range(transient_steps):
            x, Q = self.rk4(f = self.deriv, J = self.J, x = x, U = Q)

            if (i+1) % QR_steps == 0:
                            Q, R = np.linalg.qr(Q)

        Q = np.eye(4)
        for i in range(trajectory_steps):
            x, Q = self.rk4(f = self.deriv, J = self.J, x = x, U = Q)

            if (i+1) % QR_steps == 0:
                            
                Q, R = np.linalg.qr(Q)
                lambda_[:, count] = np.log(np.abs(np.diag(R))) / (dt * QR_steps)
                count += 1

        lambda_ = lambda_[:, :count]
        lyapunov_spectrum = np.mean(lambda_, axis=1)
        lyapunov_spectrum = np.sort(lyapunov_spectrum)[::-1]

        return lyapunov_spectrum
            
        

    def animate(self,
                x0):
        x = []
        x.append(x0)
        x_ = x0

        fig = plt.figure()

        ax1 = fig.add_subplot(211)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_xlim([-1*(self.l1+self.l2+0.5),(self.l1+self.l2+0.5)])
        ax1.set_ylim([-1*(self.l1+self.l2+0.5),(self.l1+self.l2+0.5)])
        pends, = ax1.plot([],[],lw=0.8)
        p1 = ax1.scatter([], [], marker='o', linewidths=3, s=100, color='r')
        p2 = ax1.scatter([], [], marker='o', linewidths=3, s=100, color='r')

        ax2 = fig.add_subplot(212)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('SVs')
        line1, = ax2.plot([],[],lw=0.8)
        line2, = ax2.plot([],[],lw=0.8)
        line3, = ax2.plot([],[],lw=0.8)
        line4, = ax2.plot([],[],lw=0.8)

        line1.set_label('SV1')
        line2.set_label('SV2')
        line3.set_label('SV3')
        line4.set_label('SV4')

        plt.ion()

        th1_l, th2_l = [],[]
        svs = []
        t = []
        count = -1
        while plt.fignum_exists(fig.number):
            count += 1
            t.append(count)
            x_, Phi = self.rk4(f = self.deriv,J=self.J, x=x_)
            theta1 = (x_[0])
            theta2 = (x_[2])
            th1_l.append(np.sin(theta1))
            th2_l.append(np.cos(theta2))

            S = np.linalg.svd(Phi, compute_uv=False)
            svs.append(S)

            x1 = self.l1*np.sin(x_[0])
            y1 = -1*self.l1*np.cos(x_[0])
            x2 = x1 + self.l2*np.sin(x_[2])
            y2 = y1 - self.l2*np.cos(x_[2])

            th1, th2 = np.asarray(th1_l), np.asarray(th2_l)
            line1.set_data(th1, th2)

            pends.set_data([0, x1, x2],
                  [0, y1, y2])
            p1.set_offsets([[x1, y1]])
            p2.set_offsets([[x2, y2]])


            if len(t) > 1000:
                t = t[-1000:]
                svs = svs[-1000:]
            
            svs_ar = np.asarray(svs)
            t_ar = np.asarray(t)
            line1.set_data(t_ar, svs_ar[:,0])
            line2.set_data(t_ar, svs_ar[:,1])
            line3.set_data(t_ar, svs_ar[:,2])
            line4.set_data(t_ar, svs_ar[:,3])

            if count % 10 == 0:
                ax1.autoscale_view()
                ax1.relim()
                ax2.set_xlim([t[-1] - 750, t[-1] + 250])
                ax2.relim()           # recompute limits from data
                ax2.autoscale_view()
                ax2.legend()
                plt.draw()
                plt.pause(0.1)
        plt.close('all')

    def find_svs(self,
                 x: np.ndarray,
                 transient_steps: int = 5000,
                 trajectory_steps: int = 1000):

        svs = []

        for _ in range(transient_steps):
            x, _ = self.rk4(self.deriv, self.J, x)

        for _ in range(trajectory_steps):
            Phi = np.eye(4)
            x, Phi = self.rk4(self.deriv, self.J, x, Phi)

            S = np.linalg.svd(Phi, compute_uv=False)
            svs.append(S)

        svs = np.asarray(svs)
        return svs


if __name__ == "__main__":
    G = 9.81
    L1 = 1.0 #Length of Pendulum1, m
    L2 = 1.0 #Length of Pendulum2, m

    L = L1 + L2
    M1 = 1.0 #Mass of Pendulum1, kg
    M2 = 1.0 #Mass of Pendulum2, kg

    dt = 0.01

    c = Double_Pendulum(M1=M1, M2=M2, L1=L1, L2=L2, G=G, dt=dt)

    theta1 = np.pi/2 #angle1, radians
    theta2 = np.pi/4#angle2, radians
    w1 = 3.0 #angular velocity 1, radians per second
    w2 = 0.0 #angular velocity 2, radians per second
    x0 = np.array([theta1, w1, theta2, w2])

    c.animate(x0)



    


    