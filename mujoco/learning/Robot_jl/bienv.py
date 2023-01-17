import gym
import random
from gym import spaces
from gym.utils import seeding
import numpy as np
from sim import Sim
from optimalize import hc,g_gene
import matplotlib.pyplot as plt
from func import matrix_to_RPY, para_period


class BipEnv(gym.Env):
    def __init__(self, Bipedal: Sim):
        self.robot = Bipedal
        self.period = 64
        self.render_flag = False
        self.jnames = ['root',
                       'Joint11', 'Joint21', 'Joint31', 'Joint32',
                       'Joint12', 'Joint22', 'Joint33', 'Joint34',
                       'Joint13', 'Joint23', 'Joint35', 'Joint36',
                       'Joint14', 'Joint24', 'Joint37', 'Joint38',
                       'Joint15', 'Joint25', 'Joint39', 'Joint310',
                       'Joint16', 'Joint26', 'Joint311', 'Joint312'
                       ]
        self.servos = ['Joint11', 'Joint12', 'Joint13', 'Joint14', 'Joint15', 'Joint16']
        self.robot.set_state(joints=len(self.jnames), jnames=self.jnames)
        # self.state = None
        self.site_pos = self.robot._data.site_xpos
        self.site_ori = self.robot._data.site_xmat
        self.n_step = 0
        self.epoch_p = 20



    # For repeatable stochasticity
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_obs(self):
        rpy = matrix_to_RPY(self.site_ori)
        return {"pos": self.site_pos, "rpy": rpy}

    def reset(self, *, seed: int = None, options: dict = None):
        self.robot.reset()
        self.n_step = 0
        return self.get_obs()

    def get_reward(self):
        pos = self.get_obs()["pos"][0]
        x, y, z = pos[0], pos[1], pos[2]
        return x

    def step(self, action):
        # Periodicity
        action = action.reshape(3, 6)
        self.n_step += 1
        period_a = para_period(para=action)
        for sid in range(self.period):
            ctrl = period_a[:, sid]
            self.robot.set_ctrl(ctrl)
            self.robot.step()
            if self.render_flag:
                self.robot.render()

        obs = self.get_obs()
        reward = self.get_reward()
        done = False
        rpy = obs["rpy"]
        # print(rpy)
        if abs(rpy[0]) > 0.6 or abs(rpy[1]) > 0.6:
            print("Turn over")
            done = True

        if self.n_step == self.epoch_p:
            done = True

        return obs, reward, done, {"done": done, "reward": reward}


if __name__ == "__main__":
    bipedal = Sim(model='./urdf/bluebody-urdf.xml', dt=0.01)
    env = BipEnv(bipedal)
    env.render_flag = True
    done_flag = False

    """opt"""
    rew0 = 0
    best_a = 0
    iter = 200
    pop_n = 20
    id = 0
    good_dis=[]
    good_pop = []
    pop = g_gene(iter, pop_n)
    for i in range(iter):
        print("gen", i)
        for j in range(pop_n):
            id +=1
            print(id)
            while not done_flag:
                obs, rew, done_flag, _ = env.step(action=pop[j])
                # if random.random() < 0.2:
                #     print(rew)
            env.reset()
            done_flag = False
    
            if rew > rew0:
                rew0 = rew
                best_a = pop[j]
                good_dis.append(rew0)
                good_pop.append(np.array(best_a).flatten())
                print(rew0)
        pop = hc(pop)
    
    np.savetxt('para_data/dis4.txt', good_dis)
    # np.savetxt('para_data/gene4.txt', good_pop)
    


    """run"""
    # a = np.array([[5.691279738097945184e-01, 6.643636018943294141e-01, 2.905038935528847510e-02,
    #                9.111924682113007323e-01, 2.107319243023748623e-01, 9.119006335388351037e-01],
    #               [8.123514608414035276e-01, 9.470319647867329049e-01, 7.006210748783945341e-01,
    #                6.488897544673272177e-01, 8.737801384380868841e-01, 7.167029661043308186e-01],
    #               [9.145012661743379123e-01, 1.074441832782022654e-01, 6.859813499997039488e-01,
    #                9.488612161944962597e-01, 3.191910588318797037e-01, 9.820748873102876919e-01]])

    # gens = np.loadtxt("para_data/gene3.txt")
    # print(gens.shape)
    
    # env.reset()
    # while not done_flag:
    #     obs, rew, done_flag, _ = env.step(action= gens[-1])
    #     print(rew)






