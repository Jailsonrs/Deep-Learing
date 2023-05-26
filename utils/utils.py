import pandas as pd
import ast
import collections
import numpy

def contar(count_list: list[int]) -> collections.defaultdict(int):
    if(isinstance(count_list, str)):
        count_list = ast.literal_eval(count_list)
    count_dict = collections.defaultdict(int)
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

import pandas as pd

a = pd.read_csv("~/Downloads/let.csv")
from collections import Counter
Counter()
a.columns
Counter(a['mês'][0])
###EXAMPLE

import itertools
for i in itertools.product(a,b):
    print(i)
a = [1,2,34,55,2,22,11,11,11,2,22,3]
b = [1,2,5,34,55,2,11,11,2,22,3]
c = [0.1,0.2,0.4,0.5]
c = {1:a,2:b}
Counter(a)
c1 = contar(c[1])
c2 = contar(c[2])
merg_dict_counts(pd.Series([c1,c2]))

et
for i in zip(a,b,c): 
    print(i)

a = {
    'a':1,'b':2,'c':3
    }

lst = ('cat', 'cat', 'dog')

for i,animal in enumerate(lst):
    print(i,animal)


ls


next(enumerate(lst)[0])
def H1(n):
    k = 1
    acc = 0
    for i in range(2,n+1):
        kn=1/i
        acc =acc+kn
    return(acc+k)

H1(10)

def Hn(n):
    k1 = [1]
    ki = [1/i for i in range(2,n+1)]
    H= k1+ki
    print(H)
    return(sum(H))
    
Hn(10)
