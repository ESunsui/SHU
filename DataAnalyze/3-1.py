import numpy as np


def get_num(list):
    aglist = []
    ctlist = []
    for item in list:
        if not item in aglist:
            aglist.append(item)
            ctlist.append(0)
        pos = aglist.index(item)
        ctlist[pos] = ctlist[pos] + 1
    return aglist, ctlist


def outputones(aglist, ctlist):
    for i in range(0, len(ctlist)):
        if ctlist[i] == 1:
            print(aglist[i])


list = [1,2,3,4,5,6,1,2,3]
outputones(get_num(list)[0], get_num(list)[1])