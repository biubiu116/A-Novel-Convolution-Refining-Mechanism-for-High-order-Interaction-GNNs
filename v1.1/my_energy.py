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
    print("Time of generating the searching dictionary: {:n} s".format(afdcal.time_genDic))
    fig, _ = afdcal.plot_dict()
    # savefig(fig, 'example_res_fast_AFD/searching_dictionary.jpg')
    # generate evaluator
    afdcal.genEva()

    print("Time of generating evaluators: {:n} s".format(afdcal.time_genEva))
    fig, _ = afdcal.plot_base_random()
    savefig(fig, 'example_res_fast_AFD/evaluator.jpg')
    # Initilize decomposition
    afdcal.init_decomp()
    print("Time of generating the searching dictionary: {:n} s".format(afdcal.time_genDic))

    # # Decomposition 10 levels
    for level in range(10):
        afdcal.nextDecomp()
        print("Total running time of the decomposition from level 1 to level {:n}: {:n} s".format(afdcal.level,sum(afdcal.run_time)))

    num = afdcal.plot_energy_rate(afdcal.level)
    return num
def total_energy(weight_list):
    total=0
    N=weight_list.shape[0]
    weight_list = weight_list.detach().numpy()
    for weight in weight_list:
        total+=energy(weight)
    return total/N