import numpy as np

from AFDCal import AFDCal
from AFDCal._io import savefig

from torch.nn.parameter import Parameter
import torch
x = np.linspace(0, 1, 1400)
#
# # 设置需要采样的信号，频率分量有200，400和600
y = 7 * np.sin(2 * np.pi * 2009 * x) + 5 * np.sin(2 * np.pi * 40 * x) + 31 * np.sin(2 * np.pi * 6040 * x)+7 * np.sin(2 * np.pi * 200 * x) + 24 * np.sin(5 * np.pi * 40 * x) + 31 * np.sin(5 * np.pi * 605 * x)
# weight_list = Parameter(torch.FloatTensor(1,10))
#
# weight_list=weight_list.detach().numpy()
# print(weight_list)
# init AFD Calculation
afdcal = AFDCal()
# Load input signal
afdcal.loadInputSignal(y)
# set decomposition method: Single Channel fast AFD
afdcal.setDecompMethod(2)
# set dictionary generation method: circle
afdcal.setDicGenMethod(2)
# generate dictionary
afdcal.genDic(1/50, 1)

print("Time of generating the searching dictionary: {:n} s".format(afdcal.time_genDic))
fig, _ = afdcal.plot_dict()
savefig(fig, 'example_res_fast_AFD/searching_dictionary.jpg')
# generate evaluator
afdcal.genEva()

print("Time of generating evaluators: {:n} s".format(afdcal.time_genEva))
fig, _ = afdcal.plot_base_random()
savefig(fig, 'example_res_fast_AFD/evaluator.jpg')
# Initilize decomposition
afdcal.init_decomp()

print("Time of decomposition at level={:n}: {:n} s".format(0,afdcal.run_time[0]))
fig, _ = afdcal.plot_decomp(0)
savefig(fig, 'example_res_fast_AFD/decomp_comp_level_{:n}.jpg'.format(0))
fig, _ = afdcal.plot_basis_comp(0)
savefig(fig, 'example_res_fast_AFD/basis_comp_level_{:n}.jpg'.format(0))
fig, _ = afdcal.plot_remainder(0)
savefig(fig, 'example_res_fast_AFD/remainder_level_{:n}.jpg'.format(0))
# Decomposition 10 levels
for level in range(10):
    afdcal.nextDecomp()    

    print("Total running time of the decomposition from level 1 to level {:n}: {:n} s".format(afdcal.level,sum(afdcal.run_time)))
    fig, _ = afdcal.plot_decomp(afdcal.level)
    savefig(fig, 'example_res_fast_AFD/decomp_comp_level_{:n}.jpg'.format(afdcal.level))
    fig, _ = afdcal.plot_basis_comp(afdcal.level)
    savefig(fig, 'example_res_fast_AFD/basis_comp_level_{:n}.jpg'.format(afdcal.level))
    fig, _ = afdcal.plot_re_sig(afdcal.level)
    savefig(fig, 'example_res_fast_AFD/reconstructed_signal_level_{:n}.jpg'.format(afdcal.level))
    fig, _ = afdcal.plot_energy_rate(afdcal.level)
    savefig(fig, 'example_res_fast_AFD/energy_convergence_rate_level_{:n}.jpg'.format(afdcal.level))
    fig, _ = afdcal.plot_searchRes(afdcal.level)
    savefig(fig, 'example_res_fast_AFD/searching_result_level_{:n}.jpg'.format(afdcal.level))
    fig, _ = afdcal.plot_remainder(afdcal.level)
    savefig(fig, 'example_res_fast_AFD/remainder_level_{:n}.jpg'.format(afdcal.level))
    fig, _ = afdcal.plot_an(afdcal.level)
    savefig(fig, 'example_res_fast_AFD/an_level_{:n}.jpg'.format(afdcal.level))