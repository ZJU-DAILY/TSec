import math

import numpy as np
import pandas as pd


class Sax:
    def __init__(self, ts, w, alpha):
        self.ts = ts
        self.w = w
        self.alpha = alpha
        self.aOffset = ord('a')  # 字符的起始位置，从a开始
        self.breakpoints = {'3': [-0.43, 0.43],
                            '4': [-0.67, 0, 0.67],
                            '5': [-0.84, -0.25, 0.25, 0.84],
                            '6': [-0.97, -0.43, 0, 0.43, 0.97],
                            '7': [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                            '8': [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],

                            }
        self.beta = self.breakpoints[str(self.alpha)]

    def normalize(self):  # 正则化
        X = np.asanyarray(self.ts)
        np.seterr(divide='ignore', invalid='ignore')
        # if np.nanstd(X) == 0:
        #     print(X)
        return (X - np.nanmean(X)) / np.nanstd(X)

    def paa_trans(self):  # 转换成paa
        tsn = self.normalize()  # 类内函数调用：法1：加self：self.normalize()   法2：加类名：SAX_trans.normalize(self)
        paa_ts = []
        n = len(tsn)
        xk = math.ceil(n / self.w)  # math.ceil()上取整，int()下取整
        for i in range(0, n, xk):
            temp_ts = tsn[i:i + xk]
            paa_ts.append(np.mean(temp_ts))
            # i = i + xk
        return paa_ts

    def to_sax(self):  # 转换成sax的字符串表示
        tsn = self.paa_trans()
        len_tsn = len(tsn)
        len_beta = len(self.beta)
        strx = ''
        for i in range(len_tsn):
            letter_found = False
            for j in range(len_beta):
                if np.isnan(tsn[i]):
                    strx += '-'
                    letter_found = True
                    break
                if tsn[i] < self.beta[j]:
                    strx += chr(self.aOffset + j)
                    letter_found = True
                    break
            if not letter_found:
                strx += chr(self.aOffset + len_beta)
        return strx

    def compare_dict(self):  # 生成距离表
        num_rep = range(self.alpha)  # 存放下标
        letters = [chr(x + self.aOffset) for x in num_rep]  # 根据alpha，确定字母的范围
        compareDict = {}
        len_letters = len(letters)
        for i in range(len_letters):
            for j in range(len_letters):
                if np.abs(num_rep[i] - num_rep[j]) <= 1:
                    compareDict[letters[i] + letters[j]] = 0
                else:
                    high_num = np.max([num_rep[i], num_rep[j]]) - 1
                    low_num = np.min([num_rep[i], num_rep[j]])
                    compareDict[letters[i] + letters[j]] = self.beta[high_num] - self.beta[low_num]
        return compareDict

    def dist(self, strx1, strx2):  # 求出两个字符串之间的mindist()距离值
        len_strx1 = len(strx1)
        len_strx2 = len(strx2)
        com_dict = self.compare_dict()

        if len_strx1 != len_strx2:
            print("The length of the two strings does not match")
        else:
            list_letter_strx1 = [x for x in strx1]
            list_letter_strx2 = [x for x in strx2]
            mindist = 0.0
            for i in range(len_strx1):
                if list_letter_strx1[i] != '-' and list_letter_strx2[i] != '-':
                    mindist += (com_dict[list_letter_strx1[i] + list_letter_strx2[i]]) ** 2
            mindist = np.sqrt((len(self.ts) * 1.0) / (self.w * 1.0)) * np.sqrt(mindist)
            return mindist


def traverse_sax(D, thread1, thread2):
    D_shape = pd.DataFrame(D).shape
    new_D = pd.DataFrame()
    for i in range(D_shape[0]):
        S = pd.DataFrame(D[i])
        D_sax = Sax(ts=S, w=thread1, alpha=thread2)
        D_i = list(D_sax.to_sax())
        D_i = pd.DataFrame(np.array(D_i)).T
        new_D = pd.concat([new_D, D_i], ignore_index=True)

    return new_D


if __name__ == '__main__':
    # 测试
    ts1 = [6.02, 6.33, 6.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 6.85, 3.85, 2.22, 1.45, 4.34,
           5.50, 1.29, 2.58, 3.83, 3.25, 6.25, 3.83, 5.63, 6.44, 6.25, 8.75, 8.83, 3.25, 0.75, 0.72]

    ts2 = [0.50, 1.29, 2.58, 3.83, 3.25, 4.25, 3.83, 5.63, 6.44, 6.25, 8.75, 8.83, 3.25, 0.75, 0.72,
           2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 6.85, 3.85, 2.22, 1.45, 4.34, ]
    x1 = Sax(ts=ts1, w=6, alpha=3)
    x2 = Sax(ts=ts2, w=6, alpha=3)
    st1 = x1.to_sax()
    st2 = x2.to_sax()
    dist = x1.dist(st1, st2)
    print('st1', st1)
    print('st2', st2)
    print(dist)
    pass
