import numpy as np

T = 24
bus_num = 27
branch_num = 31
gen_num = 8
branch_4 = np.array([[1, 4, 0.00126, 4600],
                    [19, 16, 0.000045, 3000],
                    [6, 2, 0.00049, 4800],
                    [3, 6, 0.0005, 4800],
                    [24, 21, 0.0021, 1920],
                    [16, 21, 0.00402, 2720],
                    [20, 22, 0.00048, 3600],
                    [25, 22, 0.0005, 3640],
                    [24, 22, 0.00207, 1920],
                    [16, 22, 0.00402, 2720],
                    [5, 6, 0.00108, 2920],
                    [6, 7, 0.00111, 3580],
                    [26, 23, 0.00059, 1140],
                    [27, 23, 0.00066, 1140],
                    [4, 13, 0.00521, 2720],
                    [4, 10, 0.0052, 2720],
                    [4, 5, 0.00128, 5440],
                    [10, 11, 0.00109, 4720],
                    [12, 11, 0.0007, 3520],
                    [23, 24, 0.00036, 2920],
                    [23, 16, 0.00178, 5440],
                    [10, 8, 0.00039, 5440],
                    [18, 17, 0.0007, 2900],
                    [13, 16, 0.0038, 3540],
                    [10, 16, 0.00375, 3540],
                    [17, 20, 0.00112, 2900],
                    [14, 17, 0.00056, 1140],
                    [15, 17, 0.0006, 1140],
                    [12, 5, 0.00057, 3520],
                    [9, 17, 0.00168, 1780],
                    [7, 17, 0.00166, 1780]])
branch_4[:, 3] = branch_4[:, 3]/1.5
gen_bus = np.array([1, 2, 3, 8, 9, 11, 18, 19])
load_bus = np.array([4, 5, 6, 7, 10, 12, 14, 15, 16, 17, 22, 23, 24, 26, 27])
renew_bus = np.array([20, 24, 25])
gen_pmax = np.array([1500, 1000, 1000, 800, 1000, 1200, 1000, 1500])  # 发电机最大出力
gen_pmin = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # 发电机最小出力
load_p = np.array([345, 490, 1000, 160, 960, 230, 150, 150, 425, 350, 850, 780, 760, 135, 130])  # 负荷有功需求
# load_p = (np.array([345, 490, 1000, 160, 960, 230, 150, 150, 425, 350, 850, 780, 760, 135, 130]) @
#           np.array([0.68, 0.65, 0.62, 0.60, 0.61, 0.63, 0.70, 0.69, 0.73, 0.81, 0.89, 0.92, 0.95, 0.95,
#                     0.97, 0.99, 1, 0.96, 0.96, 0.93, 0.93, 0.91, 0.77, 0.76]))  # 负荷有功需求
renew_p = np.array([1400, 2200, 2000])  # 可再生能源出力
# renew_p = np.array([[0.44, 0.702, 0.76, 0.82, 0.84, 0.84, 1, 1, 0.78, 0.64, 1, 0.92, 0.32, 0.24, 0.28, 0.3,
#                                   0.25, 0.36, 0.56, 0.84, 0.8, 0.78, 0.82, 0.52],
#                                  [0., 0., 0., 0., 0., 0.36, 0.68, 0.73, 0.85, 0.68, 0.76, 0.89, 0.96, 1., 0.95, 0.85,
#                                   0.76, 0.65, 0.29, 0., 0., 0., 0., 0.],
#                                  [0., 0., 0., 0., 0., 0.36, 0.68, 0.73, 0.85, 0.68, 0.76, 0.89, 0.96, 1., 0.95, 0.85,
#                                   0.76, 0.65, 0.29, 0., 0., 0., 0., 0.]]) * \
#                        (np.array([1400, 2200, 2000]).reshape((-1, 1)) @ np.ones((1, T)))  # 可再生能源出力
bus_load = np.zeros(bus_num)  # 节点负荷有功需求
bus_load[load_bus-1] = load_p
bus_renew = np.zeros(bus_num)  # 节点可再生能源出力
bus_renew[renew_bus-1] = renew_p
PL = sum(bus_load) - sum(bus_renew)  # 总负荷
gen_cost = np.array([[0.04, 0.025, 0.03, 0.028, 0.025, 0.024, 0.023, 0.025], [15, 14, 12, 10, 13, 9, 13, 11]]).T  # 发电成本


def case27():

    ppc = {"version": '2'}

    ppc['baseMVA'] = 1

    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = np.zeros((bus_num, 13))
    ppc["bus"][:, 0] = np.arange(1, bus_num + 1)  # 节点编号
    ppc["bus"][:, 1] = np.ones(bus_num)
    ppc["bus"][gen_bus, 1] = 2
    ppc["bus"][0, 1] = 3
    ppc["bus"][:, 2] = bus_load  # 有功负荷
    ppc["bus"][:, 6] = np.ones(bus_num)  # area
    ppc["bus"][:, 7] = np.ones(bus_num)  # Vm
    ppc["bus"][:, 9] = np.ones(bus_num)  # baseKV
    ppc["bus"][:, 10] = np.ones(bus_num)  # zone
    ppc["bus"][:, 11] = 1.05 * np.ones(bus_num)  # Vmax
    ppc["bus"][:, 12] = 0.95 * np.ones(bus_num)  # Vmin

    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = np.zeros((branch_num, 13))
    ppc["branch"][:, [0, 1, 3, 5]] = branch_4
    ppc["branch"][:, 6] = ppc["branch"][:, 5]
    ppc["branch"][:, 7] = ppc["branch"][:, 5]
    ppc["branch"][:, 10] = np.ones(branch_num)
    ppc["branch"][:, 11] = -360 * np.ones(branch_num)
    ppc["branch"][:, 12] = 360 * np.ones(branch_num)

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = np.zeros((gen_num, 21))
    ppc["gen"][:, 0] = gen_bus
    ppc["gen"][:, 5] = np.ones(gen_num)
    ppc["gen"][:, 6] = 100 * np.ones(gen_num)
    ppc["gen"][:, 7] = np.ones(gen_num)
    ppc["gen"][:, 8] = gen_pmax
    ppc["gen"][:, 9] = gen_pmin

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = np.zeros((gen_num, 7))
    ppc["gencost"][:, 4:6] = gen_cost

    return ppc
