import pandas as pd
import torch
import numpy as np
from lorenz import LorenzGenerator
import random
from typing import Optional


#REPRODUCABILITY

class traj_Dataset(torch.utils.data.Dataset):

    '''
    Dataset containing pairs of (1,3) points and their following output. All z-score normalised accoridng to input mean, std , or if empty, clauclated from inputs.
    One can also load a presvaed dataset.
    
    '''
    def __init__(self,
                 n_trajectories: int,
                 n_samples_per_traj: int,
                 n_transient: int,
                 dt: int = 0.01, 
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,
                 preloaded: Optional[dict] = None,
                 RANDOM_SEED = random.randint(1,100)):
        '''
        Inputs:
            n_trajectories (int): The amount of initnial starting points of trajectories to use.
            n_samples_per_traj (int): As it sounds.
            n_transient (int): Number of trajectories to throw away at the start to ensure samples are on the attractor.

        '''

        
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.use_deterministic_algorithms(True)
        self.traj_generator = LorenzGenerator()

        self.n_trajectories = n_trajectories
        self.n_samples_per_traj = n_samples_per_traj
        self.n_transient = n_transient
        self.dt = dt
        if preloaded is not None:
            # Use saved data directly
            self.samples = preloaded['samples']
            self.targets = preloaded['targets']
            self.mean = preloaded['mean']
            self.std = preloaded['std']
        else:

            self.samples, self.targets = self.generate_samples()

            if mean == None:
                self.mean =torch.mean(self.samples, axis=0)
                self.std =torch.std(self.samples, axis=0)
            else:
                self.mean = mean
                self.std = std
            
            self.samples = (self.samples - self.mean) / self.std
            self.targets = (self.targets - self.mean) / self.std

        print(f"Initialised Dataset:\n{self.n_trajectories} Trajectories \n{self.n_samples_per_traj} Samples per Trajectory\n{self.n_transient} Transient steps\nh = {self.dt}")

    def generate_samples(self):
        samples = np.empty((0,3))
        targets = np.empty((0,3))

        for i in range(int(self.n_trajectories*0.5)):

            x0 = np.array([np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0,50)])
            traj = self.traj_generator.generate_trajectory(x0 = x0,
                                                    n_steps = (self.n_samples_per_traj + self.n_transient),
                                                    dt = self.dt)
            traj = traj[self.n_transient+1:]

            samples = np.vstack([samples, traj])

            last_target = self.traj_generator.rk4(self.traj_generator.calc_derivatives, x = traj[-1], dt = self.dt)

            targets = np.vstack([targets,np.vstack([traj[1:], last_target])])

        
        for i in range(int(self.n_trajectories*0.5)):
            
            point = np.array([8.485, 8.485, 27]) if i % 2 == 0 else np.array([-8.485, -8.485, 27])            
            directions = np.random.normal(size=3)
            directions /= np.linalg.norm(directions)

            # random radius with correct volume distribution
            r = 5 * np.random.rand()**(1/3)

            point = point + r * directions
            traj = self.traj_generator.generate_trajectory(x0 = point,
                                                    n_steps = (self.n_samples_per_traj+ self.n_transient),
                                                    dt = self.dt)
            traj = traj[self.n_transient+1:]


            samples = np.vstack([samples, traj])

            last_target = self.traj_generator.rk4(self.traj_generator.calc_derivatives, x = traj[-1], dt = self.dt)

            targets = np.vstack([targets,np.vstack([traj[1:], last_target])])
        samples = torch.tensor(samples)
        targets = torch.tensor(targets)

        return samples, targets 
        


    def __len__(self):
        return (len(self.samples))

    def __getitem__(self, idx):

        return self.samples[idx], self.targets[idx]
    
if __name__ == '__main__':
    # train_set = torch.load(r".\dataset\small_train_set_dataset.pt")
    # mean = train_set['mean']
    # std = train_set['std']
    # dataset = traj_Dataset(n_trajectories=12, n_samples_per_traj=5, n_transient = 5000, mean = mean, std = std)
    # torch.save({
    #     'samples': dataset.samples,
    #     'targets': dataset.targets,
    #     'mean': dataset.mean,
    #     'std': dataset.std
    # }, r'.\dataset\small_val_set_dataset.pt')
    val_set = torch.load(r".\dataset\small_val_set_dataset.pt")
    test_set = torch.load(r".\dataset\small_test_set_dataset.pt")

    print(test_set['samples'])
    print(val_set['samples'])






