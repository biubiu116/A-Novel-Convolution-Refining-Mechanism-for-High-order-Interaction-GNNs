import numpy as np

from AFDCal import AFDCal
from AFDCal._io import savefig

from torch.nn.parameter import Parameter
import torch

def energy(weight):

    afdcal = AFDCal()
    # Load input signal
    afdcal.loadInputSignal(weight)
    # set decomposition method: Single Channel fast AFD
    afdcal.setDecompMethod(2)
    # set dictionary generation method: circle
    afdcal.setDicGenMethod(2)
    # generate dictionary
    afdcal.genDic(1/50, 1)
    # print("Time of generating the searching dictionary: {:n} s".format(afdcal.time_genDic))
    # fig, _ = afdcal.plot_dict()
    # savefig(fig, 'example_res_fast_AFD/searching_dictionary.jpg')
    # generate evaluator
    afdcal.genEva()

    # print("Time of generating evaluators: {:n} s".format(afdcal.time_genEva))
    # fig, _ = afdcal.plot_base_random()
    # savefig(fig, 'example_res_fast_AFD/evaluator.jpg')
    # Initilize decomposition
    afdcal.init_decomp()
    # print("Time of generating the searching dictionary: {:n} s".format(afdcal.time_genDic))

    # # Decomposition 10 levels
    for level in range(10):
        afdcal.nextDecomp()
        # print("Total running time of the decomposition from level 1 to level {:n}: {:n} s".format(afdcal.level,sum(afdcal.run_time)))

    num = afdcal.plot_energy_rate(afdcal.level)
    return num
def total_energy(weight_list):
    total=0
    N=weight_list.shape[0]
    # N =8
    weight_list = weight_list.detach().cpu().numpy()
    weight_list=weight_list.reshape(1,-1)
    for weight in weight_list:
        t=energy(weight)
        total+=t
        # print(t)
    return total/N
# weight_list = np.arange(320).reshape(1,320)
# weight_list=np.array([[ 0.0536, -0.0968, -0.0637, -0.2085,  0.0728,  0.1681,  0.0988, -0.0917,
#           0.0605,  0.0036,  0.0783,  0.0351, -0.0920,  0.0501, -0.0461, -0.2118,
#           0.0226, -0.0750,  0.0702,  0.0924,  0.1488,  0.1401,  0.0572,  0.0230,
#           0.0153, -0.0298,  0.1737,  0.0594, -0.0862,  0.0216,  0.0350,  0.0019]])
# # print(weight_list)
# print(total_energy(weight_list))