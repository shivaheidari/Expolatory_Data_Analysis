import pandas as pd
import numpy as np
import scipy.stats as stats
# class read_file:
#     def __init__(self, data_file, setting):
#         self.data = pd.read_csv(data_file)
#         self.df = pd.DataFrame(self.data)
#         self.setting =

class chi_square:

    def __init__(self, var1, var2, prob):
        self.var1 = var1
        self.var2 = var2
        self.prob = prob
        self.contigency_matrix = 0
        self.dict_vals = {"chi_stat": 0, "p_val" :0, "dof": 0, "expected":0}

    def significance(self):
        contigency_matrix = pd.crosstab(self.var1, self.var2)
        self.contigency_matrix = contigency_matrix
        chi_stat, p_val, dof, expected = stats.chi2_contingency(contigency_matrix)
        self.dict_vals["chi_stat"] = chi_stat
        self.dict_vals["p_val"] = p_val
        self.dict_vals["dof"] = dof
        self.dict_vals["expected"] = expected
        critical = stats.chi2.ppf(self.prob, dof)
        if abs(chi_stat) >= critical:
            return "reject H0"
        else:
            return "Fail to reject"

    def get_contigency_matrix(self):
        return self.contigency_matrix

    def get_total(self):
        return self.dict_vals
    def print_to_file(self):
        file = open("chi_export.txt", 'w+')
        file.write(str(self.dict_vals)+"\n"+str(self.contigency_matrix))
        file.close()


data = pd.read_csv("dt_test.csv")
df = pd.DataFrame(data)
ob = chi_square(df["v1"], df["v2"], 0.95)
print(ob.significance())
ob.print_to_file()
# print(ob.get_total())
# print(ob.get_contigency_matrix())
