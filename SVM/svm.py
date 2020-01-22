from numpy import *
import random
from time import *


def loadDataSet(fileName):  # 输出dataArr(m*n),labelArr(1*m)其中m为数据集的个数
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')  # 去除制表符，将数据分开
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 数组矩阵
        labelMat.append(float(lineArr[2]))  # 标签
    return dataMat, labelMat


def selectJrand(i, m):  # 随机找一个和i不同的j
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):  # 调整大于H或小于L的alpha的值
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn);
    labelMat = mat(classLabels).transpose()  # 转置
    b = 0;
    m, n = shape(dataMatrix)  # m为输入数据的个数，n为输入向量的维数
    alpha = mat(zeros((m, 1)))  # 初始化参数，确定m个alpha
    iter = 0  # 用于计算迭代次数
    while (iter < maxIter):  # 当迭代次数小于最大迭代次数时（外循环）
        alphaPairsChanged = 0  # 初始化alpha的改变量为0
        for i in range(m):  # 内循环
            fXi = float(multiply(alpha, labelMat).T * \
                        (dataMatrix * dataMatrix[i, :].T)) + b  # 计算f(xi)
            Ei = fXi - float(labelMat[i])  # 计算f(xi)与标签之间的误差
            if ((labelMat[i] * Ei < -toler) and (alpha[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alpha[i] > 0)):  # 如果可以进行优化
                j = selectJrand(i, m)  # 随机选择一个j与i配对
                fXj = float(multiply(alpha, labelMat).T * \
                            (dataMatrix * dataMatrix[j, :].T)) + b  # 计算f(xj)
                Ej = fXj - float(labelMat[j])  # 计算j的误差
                alphaIold = alpha[i].copy()  # 保存原来的alpha(i)
                alphaJold = alpha[j].copy()
                if (labelMat[i] != labelMat[j]):  # 保证alpha在0到c之间
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H: print('L=H');continue
                eta = 2 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0: print('eta=0');continue
                alpha[j] -= labelMat[j] * (Ei - Ej) / eta
                alpha[j] = clipAlpha(alpha[j], H, L)  # 调整大于H或小于L的alpha
                if (abs(alpha[j] - alphaJold) < 0.0001):
                    print('j not move enough');
                    continue
                alpha[i] += labelMat[j] * labelMat[i] * (alphaJold - alpha[j])
                b1 = b - Ei - labelMat[i] * (alpha[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alpha[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T  # 设置b
                b2 = b - Ej - labelMat[i] * (alpha[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alpha[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alpha[i]) and (C > alpha[j]):
                    b = b1
                elif (0 < alpha[j]) and (C > alpha[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphaPairsChanged += 1
                print('iter:%d i:%d,pairs changed%d' % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteraction number:%d' % iter)
    return b, alpha


# 定义径向基函数
def kernelTrans(X, A, kTup):  # 定义核转换函数（径向基函数）
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # 线性核K为m*1的矩阵
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print("L==H"); return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (
                alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# smoP函数用于计算超平的alpha,b
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # 完整的Platter SMO
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0  # 计算循环的次数
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


# calcWs用于计算权重值w
def calcWs(alphas, dataArr, classLabels):  # 计算权重W
    X = mat(dataArr);
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


# 值得注意的是测试准确与k1和C的取值有关。
def testRbf(k1=1.3):  # 给定输入参数K1
    # 测试训练集上的准确率
    dataArr, labelArr = loadDataSet('testSetRBF.txt')  # 导入数据作为训练集
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]  # 找出alphas中大于0的元素的位置
    # 此处需要说明一下alphas.A的含义
    sVs = datMat[svInd]  # 获取支持向量的矩阵，因为只要alpha中不等于0的元素都是支持向量
    labelSV = labelMat[svInd]  # 支持向量的标签
    print("there are %d Support Vectors" % shape(sVs)[0])  # 输出有多少个支持向量
    m, n = shape(datMat)  # 数据组的矩阵形状表示为有m个数据，数据维数为n
    errorCount = 0  # 计算错误的个数
    for i in range(m):  # 开始分类，是函数的核心
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))  # 计算原数据集中各元素的核值
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b  # 计算预测结果y的值
        if sign(predict) != sign(labelArr[i]): errorCount += 1  # 利用符号判断类别
        ### sign（a）为符号函数：若a>0则输出1，若a<0则输出-1.###
    print("the training error rate is: %f" % (float(errorCount) / m))
    # 2、测试测试集上的准确率
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)  # labelMat = mat(labelArr).transpose()此处可以不用
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


def main():
    t1 = time()
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.01, 40)
    ws = calcWs(alphas, dataArr, labelArr)
    testRbf()
    t2 = time()
    print("程序所用时间为%ss" % (t2 - t1))


if __name__ == '__main__':
    main()
