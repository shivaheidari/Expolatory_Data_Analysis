import pandas as pd
import numpy as np
import json
import scipy.stats as stats
# class read_file:
#     def __init__(self, data_file, setting):
#         self.data = pd.read_csv(data_file)
#         self.df = pd.DataFrame(self.data)
#         self.setting =


class file_reader:
    def __init__(self):

        self.data = pd.read_csv("dt_test.csv")
        self.df = pd.DataFrame(self.data)

        with open("setting.json") as file:
            setting = json.load(file)

        self.data_type = setting["data_type"]
        self.confidence = setting["confidance"]
        self.source = setting["variable_source"]
        self.target = setting["variable_target"]
    def create_test(self):
        if self.data_type == "categorical":
            for s in self.source:
                for t in self.target:
                    obj = chi_square(self.df[s], self.df[t], self.confidence)
                    obj.significance()
                    obj.print_to_file()





class chi_square:

    def __init__(self, var1, var2, prob):
        self.var1 = var1
        self.var2 = var2
        self.prob = prob
        self.contigency_matrix = 0
        self.res = " "
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
            self.res = "reject H0"

        else:
            self.res = "Fail to reject"
        return self.res

    def get_contigency_matrix(self):
        return self.contigency_matrix

    def get_total(self):
        return self.dict_vals
    def print_to_file(self):
        file = open("chi_export.txt", 'w+')
        file.write(str(self.var1)+"and"+str(self.var2)+self.res)
        file.write(str(self.dict_vals)+"\n"+str(self.contigency_matrix))
        file.close()




obj = file_reader()
obj.create_test()
# data = pd.read_csv("dt_test.csv")
# df = pd.DataFrame(data)
# ob = chi_square(df["v1"], df["v2"], 0.95)
# print(ob.significance())
# ob.print_to_file()
# print(ob.get_total())
# print(ob.get_contigency_matrix())
