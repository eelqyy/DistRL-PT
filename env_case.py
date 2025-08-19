import numpy as np
from case_data import case27
from case_data import gen_cost
import pandas as pd
from docplex.mp.model import Model
from pypower.idx_bus import PD
from pypower.idx_gen import PMAX, PMIN, GEN_BUS
from pypower.ext2int import ext2int
import copy


class Case27:

    def __init__(self):
        self.T = 24
        self.strategy_id = [0]
        self.producer_num = len(self.strategy_id)
        self.state_size = self.T
        self.action_size = 20
        self.max_gen = np.array([800, 1000, 1000, 1500, 1000, 1200, 1000, 1500]).reshape(-1, 1)
        self.max_bid = gen_cost[:, 1].reshape(-1, 1) + 5
        self.min_bid = gen_cost[:, 1].reshape(-1, 1)
        self.max_action = 1.5
        self.min_action = 1.0
        self.re_fluctuation = 0.005
        self.re_ratio = 0.01
        self.pv_gen_base = \
            np.array([0.04106311, 0.04109686, 0.04116128, 0.04537202, 0.04748286, 0.04066248,
                      0.05284677, 0.14450161, 0.31243502, 0.42640106, 0.80315431, 0.93181193,
                      0.84293841, 0.86374954, 0.79546408, 1.,         0.91369019, 0.58135568,
                      0.3126375,  0.06166072, 0.04202567, 0.04142042, 0.04158805, 0.04126079])
        self.wt_gen_base = \
            np.array([0.88670299, 0.9093538,  0.89311449, 0.92751207, 0.95298483, 0.87214335,
                      0.84727164, 0.89891893, 0.88069668, 0.92800528, 0.78828146, 0.74671535,
                      0.65405882, 0.51920542, 0.54432076, 0.57250675, 0.57579582, 0.86749372,
                      0.89611113, 0.99450444, 1.,         0.99853876, 0.94160415, 0.94547958])
        self.load_base = \
            np.array([0.795209787, 0.741720839, 0.700496665, 0.672710131, 0.66423111, 0.675798927,
                      0.703702259, 0.7443024, 0.810712211, 0.8798551, 0.924775688, 0.917510485,
                      0.896827839, 0.887136375, 0.877445168, 0.868360892, 0.897254289, 0.938112001,
                      0.988058029, 0.99481591, 0.999317611, 1, 0.962562996, 0.904003575])
        self.power_sum = 10000
        self.load_para = np.array([0, 300, 300, 400, 0]) * 10
        self.renew_para = np.array([100, 150]) * self.power_sum * self.re_ratio / 250

    def RT_market_clear(self, actions):
        T = self.T
        ppc = case27()
        ppc = ext2int(ppc)
        baseMVA = ppc["baseMVA"]
        branch = ppc["branch"]
        bus = ppc["bus"]
        gen = ppc["gen"]
        gencost = ppc["gencost"]
        branch_num = branch[:, 0].size
        bus_num = bus[:, 0].size
        gen_num = gen[:, 0].size
        gen_bus = gen[:, GEN_BUS]
        gen_pmax = gen[:, PMAX] / baseMVA
        gen_pmax = gen_pmax / sum(gen_pmax) * self.power_sum * (1 - self.re_ratio * (1 - 2 * self.re_fluctuation))
        gen_pmin = gen[:, PMIN] / baseMVA
        bus_load = bus[:, PD] / baseMVA
        pv_gen = (self.pv_gen_base * self.renew_para[0]) * (
                    1 - 2 * self.re_fluctuation + self.re_fluctuation * np.random.rand(1)) / baseMVA
        wt_gen = (self.wt_gen_base * self.renew_para[1]) * (
                    1 - 2 * self.re_fluctuation + self.re_fluctuation * np.random.rand(1)) / baseMVA
        load = (self.load_base * sum(self.load_para)) * (
                    1 - 2 * self.re_fluctuation + self.re_fluctuation * np.random.rand(1)) / baseMVA
        PL = (load - pv_gen - wt_gen)
        para = copy.deepcopy(gencost[:, 4:7])
        para = np.repeat(para[np.newaxis, ...], self.T, axis=0)
        para[:, self.strategy_id, 1] *= actions.reshape(self.T, 1)
        model = Model()
        pg = [[model.continuous_var(name='x_' + str(i) + '_' + str(j)) for j in range(T)] for i in
              range(gen_num)]
        totalcost = 0
        for i in range(gen_num):
            for j in range(T):
                totalcost += (0.5 * para[j, i, 0] * pg[i][j] * baseMVA + para[j, i, 1]) * pg[i][j] * baseMVA
        for j in range(T):
            model.add_constraint(PL[j] == list(map(sum, zip(*pg)))[j], 'P_demand')
        for i in range(gen_num):
            model.add_constraints(pg_i <= gen_pmax[i] for pg_i in pg[i])
            model.add_constraints(pg_i >= gen_pmin[i] for pg_i in pg[i])
        model.minimize(totalcost)
        sol = model.solve()
        lmp = model.dual_values(model.find_matching_linear_constraints('P_demand'))
        output = np.zeros((gen_num, T))
        for i in range(gen_num):
            output[i] = sol.get_values(pg[i])
        profit = np.ones((gen_num, T))
        for i in range(gen_num):
            for j in range(T):
                income = output[i][j] * lmp[j]
                gencost_qua = (0.5 * gencost[i, 4] * output[i][j] * baseMVA + gencost[i, 5]) * output[i][j] * baseMVA
                profit[i, j] = income - gencost_qua

        return output, lmp, profit

    def action_real(self, actions):
        actions = np.array(actions) * (self.max_action - self.min_action) / (self.action_size - 1) + self.min_action
        return actions

    def step_forward(self, actions):
        actions = self.action_real(actions)
        output, lmp, profit = self.RT_market_clear(actions)
        reward = profit[self.strategy_id] / 1000
        return reward
