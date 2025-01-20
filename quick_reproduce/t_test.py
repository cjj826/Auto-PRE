import numpy as np
from scipy.stats import ttest_rel
import json

def t_test(A, B):
    sampleA = np.asarray(A)
    sampleB = np.asarray(B)
    r = ttest_rel(sampleA, sampleB)

    print("statistic:", r.__getattribute__("statistic"))
    print("pvalue:", r.__getattribute__("pvalue")) 
    # pvalue远小于0.05，认为两样本均值存在显著差异
    # 小于 0.01 差异更小

if __name__ == '__main__':

    tasks = ['Xsum', 'NF_CATS', 'Dialog']
    # modes = ['5-level', '100-level', 'pairwise']
    # modes = ['5-level', '100-level']
    modes = ['5level', '100level', 'pairwise']
    # modes = ['pairwise']
    ms = ['cps', 'g', 'wo']

    task = tasks[2]

    for mode in modes:
        for m in ms:
            print(m, mode)
            gpt4_path = f"./significant_test/{task}/gpt4_{mode}.json"
            base_path = f"./significant_test/{task}/{m}_{mode}.json"

            path1 = gpt4_path
            path2 = base_path

            A = json.loads(open(path1, "r").readlines()[0])["scores"]
            B = json.loads(open(path2, "r").readlines()[0])["scores"]
            t_test(A, B)

            print()