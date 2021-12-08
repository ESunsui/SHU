import numpy as np
import math

orilist = [1,2,3,4,5,6,7,8,9,10,11,12]
num_div = 3
div_len = math.ceil(len(orilist)/num_div)


def _div(listTemp, div_len):
    for i in range(0, len(listTemp), div_len):
        yield listTemp[i:i + div_len]

def div(listTemp, div_len):
    rt = []
    temp = _div(listTemp, div_len)
    for item in temp:
        rt.append(item)
    return rt


def sort(llist):
    for iter in llist:
        iter.sort()


def get_avg(list, i):
    sum = 0
    for item in list:
        sum = sum + item
    return sum/i


def get_closest_err(list, avg):
    asum = 0
    for iter in list:
        asum += iter
    diff = asum - avg
    if asum > avg:
        dis = abs(diff - list[0])
        pos = 0
        apos = 0
        for iter in list:
            if abs(diff - iter) < dis:
                apos = pos
                dis = abs(diff - iter)
            pos += 1
        return diff-list[apos], list.pop(apos)
    else:
        return diff-list[0], list.pop(0)


def insert_closest(num, list):
    dst = abs(num - list[0])
    rpos = 0
    pos = 0
    for item in list:
        if abs(num - item) < dst:
            dst = abs(num - item)
            rpos = pos
        pos += 1
    return list.pop(rpos)


def update(llist, avg):
    llist.append(llist.pop(0))
    upd = []
    spare = []
    for list in llist:
        los, num = get_closest_err(list, avg)
        upd.append(los)
        spare.append(num)
    while len(spare)>0:
        count = 0
        for i in upd:
            if i<0 and len(spare)>0:
                llist[count].append(insert_closest(i, spare))
                llist[count].sort()
            count += 1


def proc():
    llist = div(orilist, div_len)
    sort(llist)
    print(llist)
    avg = get_avg(orilist, num_div)
    print(avg)
    for i in range(0, 20):
        update(llist, avg)
        print(llist)

proc()