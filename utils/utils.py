import pandas as pd
import ast
import collections
import numpy

def contar(count_list: list[int]) -> collections.defaultdict(int):
    count_dict = collections.defaultdict(int)
    if(isinstance(count_list, str)):
        count_list = ast.literal_eval(count_list)
    count_list.pop(0)
    for i in count_list:
        if i in count_dict:
            count_dict[i] += 1
        else:
            count_dict[i] = 1
    return count_dict

def merg_dict_counts(counter: pd.Series) -> collections.defaultdict(int):
    couter_dict  = counter.to_dict()
    merged_data = collections.defaultdict(int)
    for value in couter_dict.values():
        for key, count in value.items():
            merged_data[key] += count
    return(merged_data)


###EXAMPLE
a = [1,2,34,55,2,22,11,11,11,2,22,3]
b = [1,2,5,34,55,2,11,11,2,22,3]
c = {1:a,2:b}
c1 = contar(c[1])
c2 = contar(c[2])
merg_dict_counts(pd.Series([c1,c2]))

