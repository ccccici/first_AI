import os
import time
import random
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import ogr


def cal_time(secs):
    minute = secs // 60
    second = secs % 60
    hour = minute // 60
    minute = minute % 60
    return hour, minute, second


def create_dir(path):
    abs_path = os.getcwd()
    if not os.path.exists(path):  # 创建操作目录
        os.mkdir(path)
        out_path = path
    else:
        out_path = path
        print("{} is already exist!".format(path), end="")
        time.sleep(1)
        print("\r ", end="")
    return os.path.join(abs_path, out_path)


class Utils:
    @staticmethod
    def is_dominate(obj_a, obj_b, num_obj):  # a是否完全支配b
        if type(obj_a) is not np.ndarray:
            obj_a, obj_b = np.array(obj_a), np.array(obj_b)
        res = np.array([np.sign(k) for k in obj_a - obj_b])    # 取符号
        res_ngt0, res_eqf1 = np.argwhere(res <= 0), np.argwhere(res == -1)   # 返回所有（）内条件的索引值
        if res_ngt0.shape[0] == num_obj and res_eqf1.shape[0] > 0:
            return True
        return False


class Pareto(object):
    def __init__(self, pop_obj):
        self.pop_size = pop_obj.shape[0]   # 行数
        self.pop_obj = pop_obj
        self.num_obj = pop_obj.shape[1]   # 列数
        self.f_ = []
        self.sp = [[] for _ in range(self.pop_size)]  # 种群内每个抗体所完全支配的抗体
        self.np = np.zeros([self.pop_size, 1], dtype=int)  # 种群内每个抗体被支配次数
        self.rank = np.zeros([self.pop_size, 1], dtype=int)   # 支配次数为0的抗体
        self.cd = np.zeros([self.pop_size, 1])
        self.fast_non_dominate_sort()
        self.crowd_distance()

    def __index(self, i):  # 不与自身进行支配判断
        return np.delete(range(self.pop_size), i)

    def __is_dominate(self, i, j):   # 判断是否i完全支配j，返回的是true或者false值
        return Utils.is_dominate(self.pop_obj[i], self.pop_obj[j], self.num_obj)

    def __f1_dominate(self):  # 寻找被支配次数为0的抗体
        f1 = []
        for i in range(self.pop_size):
            for j in self.__index(i):
                if self.__is_dominate(i, j):  # i支配j
                    if j not in self.sp[i]:  # i支配j 则将j放入i所支配抗体
                        self.sp[i].append(j)
                elif self.__is_dominate(j, i):  # 否则j支配i  i被支配次数加1
                    self.np[i] += 1
            if self.np[i] == 0:
                self.rank[i] = 1
                f1.append(i)
        return f1

    def fast_non_dominate_sort(self):  # 快速非支配等级排序
        rank = 1
        f1 = self.__f1_dominate()
        while f1:
            self.f_.append(f1)
            q = []
            for i in f1:
                for j in self.sp[i]:
                    self.np[j] -= 1
                    if self.np[j] == 0:
                        self.rank[j] = rank + 1
                        q.append(j)
            rank += 1
            f1 = q

    def sort_obj_by(self, f_=None, j=0):
        if f_ is not None:
            index = np.argsort(self.pop_obj[f_, j])
        else:
            index = np.argsort(self.pop_obj[:, j])
        return index

    def crowd_distance(self):  # 计算拥挤距离
        for f_ in self.f_:
            len_f1 = len(f_) - 1
            for j in range(self.num_obj):
                index = self.sort_obj_by(f_, j)
                sorted_obj = self.pop_obj[f_][index]
                obj_range_fj = sorted_obj[-1, j] - sorted_obj[0, j]
                self.cd[f_[index[0]]] = np.inf
                self.cd[f_[index[-1]]] = np.inf
                for i in f_:
                    k = np.argwhere(np.array(f_)[index] == i)[:, 0][0]
                    if 0 < index[k] < len_f1:
                        self.cd[i] += (sorted_obj[index[k] + 1, j] - sorted_obj[index[k] - 1, j]) / obj_range_fj

    def elite_strategy(self):  # 精英策略排序
        new_args = []
        for arg in self.f_:
            cd = self.cd[arg]
            args_ = np.argsort(-cd.flatten())
            new_args.append([arg[i] for i in args_])
        args = list(itertools.chain.from_iterable(new_args))  # 非支配等级内拥挤距离降序
        self.f_ = [len(i) for i in self.f_]
        self.rank = self.rank[args]
        self.cd = self.cd[args]
        return args


class MissionSplit:
    @staticmethod
    def parallel_split(f_, core):
        N = len(f_)
        if N <= core:
            return [[i] for i in f_]
        else:
            nf_ = []
            a = N % core
            b = N // core
            if a != 0:
                for i in range(0, N - a, b):
                    nf_.append(f_[i:i + b])
                for i, _f_ in enumerate(nf_[:a]):
                    _f_.append(f_[-(i + 1)])
            else:
                for i in range(0, N, b):
                    nf_.append(f_[i:i + b])
            return nf_


class ObjFunc:
    def __init__(self, OD_array, core):
        """
        :param OD_array: arcgisOD成本矩阵求解结果
        :param core: 调用cpu核数
        """
        self.ms = MissionSplit()
        self.OD_array = OD_array
        if core > os.cpu_count():
            core = os.cpu_count()
        self.core = core

    def __get_demand(self, X_):
        target = self.OD_array[np.isin(self.OD_array[:, 0], X_)]
        p2id = np.unique(target[:, 1])
        new_t = np.zeros((len(p2id), 3))
        for i, pid in enumerate(p2id):
            new_t[i] = target[target[:, 1] == pid][0, -3:]
        return new_t

    def parallel_func(self, X):  # 定义目标函数-并行
        result_ = np.zeros((len(X), 3))
        for i, X in enumerate(X):
            result_[i] = self.func(X)
        return result_

    def func(self, X):  # 定义目标函数-单抗体
        X_ = self.__get_demand(X)
        pnum = sum(X_[:, 1])
        result1 = sum(X_[:, 0] * X_[:, 1]) / pnum  #目标1
        result2 = 1 / pnum #目标2
        result3 = sum(X_[:, 2])  #目标3
        return [result1, result2, 1 / result3]

    def parallel_cal(self, f_):
        f_num = self.ms.parallel_split([i for i in range(len(f_))], self.core)
        f_num = list(itertools.chain.from_iterable(f_num))
        args = np.argsort(f_num)
        f_ = self.ms.parallel_split(f_, self.core)

        pool = multiprocessing.Pool(self.core)
        MSLL_ = pool.map(self.parallel_func, f_)
        pool.close()
        MSLL_ = np.vstack(MSLL_)
        return MSLL_[args]


class IASS(object):
    def __init__(self, func, site, pop, Gen, NS, NP, Ncl, max_PF, pm):
        """
        :param func: 目标函数
        :param site: 待选点
        :param pop: 需求点
        :param Gen: 最大迭代次数
        :param NS: 选址个数
        :param NP: 种群数量
        :param Ncl: 克隆个数
        :param max_PF: 最大非支配解个数(建议取值20-40)
        :param pm: 调节变异率取值的常量（建议取值0.25-5）
        """
        self.func = func
        self.site = site
        self.pool = set(self.func.OD_array[:, 0])  # 待选点池
        self.pop = pop

        self.Gen = Gen
        self.gen = 0  # 当前迭代次数
        self.NS = NS
        self.NP = NP
        self.Ncl = Ncl
        self.max_PF = max_PF
        self.pm = pm
        self.del_indiv = []

        self.initial_time = time.time()
        f_ = self.__pop_initial(self.NP)
        self.f, self.MSLL, _ = self.__evaluate(f_)

    def __pop_initial(self, pop_num):  # 初始化种群（抗体）
        f_ = []
        for i in range(pop_num):
            _f_ = random.sample(self.pool, self.NS)
            while set(_f_) in self.del_indiv:
                _f_ = random.sample(self.pool, self.NS)
            f_.append(_f_)
            self.del_indiv.append(set(_f_))
        return f_

    def __evaluate(self, f_):  # 抗体质量评价
        MSLL_ = self.func.parallel_cal(f_)
        pareto = Pareto(MSLL_)
        args = pareto.elite_strategy()  # 精英策略排序索引
        f_ = [f_[i] for i in args]
        return f_[:pareto.f_[0]], MSLL_[args][:pareto.f_[0]], MSLL_  # 非支配种群

    def __clone(self, individual):  # 克隆|变异
        Naf = []
        pm = 1 / self.NS * self.pm * np.exp((1 - self.gen / self.Gen) ** 2)

        for i in range(self.Ncl):
            indiv = individual.copy()
            pn = 0
            for j in range(self.NS):
                if np.random.random(1)[0] < pm:  # 变异
                    new_j = random.sample(self.pool, 1)
                    indiv[j] = new_j[0]
                    while set(indiv) in self.del_indiv:
                        new_j = random.sample(self.pool, 1)
                        indiv[j] = new_j[0]
                    self.del_indiv.append(set(indiv))
                    pn += 1
            if pn > 0:
                Naf.append(indiv)
        return Naf

    def __immune(self):  # 免疫
        af = []
        for i in range(len(self.f)):
            individual = self.f[i]
            Naf = self.__clone(individual)
            af += Naf
        return af

    def main(self, out_data_path):  # 免疫
        trace_ = []
        MSLL_ = []
        th_num = 0

        plt.ion()
        fig = plt.figure(figsize=(30, 20))
        while self.gen < self.Gen:
            iter_time = time.time()
            af = self.__immune()
            af, _, _ = self.__evaluate(af + self.f)
            if len(af) > self.max_PF:  # 保持非支配解数量
                af = af[:self.max_PF]

            bf = self.__pop_initial(self.NP - len(af))  # 补充抗体
            self.f, MSLL_, MSLL_pop = self.__evaluate(af + bf)
            if len(self.f) > self.max_PF:  # 保持非支配解数量
                self.f = self.f[:self.max_PF]
                MSLL_ = MSLL_[:self.max_PF]
            self.gen += 1

            if np.all(np.average(MSLL_, axis=0) >= np.average(self.MSLL, axis=0)):
                th_num += 1
            else:
                th_num = 0

            trace_.append([np.average(MSLL_pop, axis=0), np.average(MSLL_, axis=0)])
            self.plot(fig, self.f[0], MSLL_, self.MSLL, np.array(trace_))
            self.MSLL = MSLL_
            print("\rIteration: {} | Spend: {:.2f} secs.".format(self.gen, time.time()-iter_time), end="")

            if th_num >= 5:
                self.plot(fig, self.f[0], MSLL_, self.MSLL, np.array(trace_))
                plt.savefig(os.path.join(out_data_path, "Result of multi_obj_opt.png"))
                plt.ioff()
                plt.close()
                return self.f, self.MSLL, np.array(trace_)

        self.plot(fig, self.f[0], MSLL_, self.MSLL, np.array(trace_))
        plt.savefig(os.path.join(out_data_path, "Result of multi_obj_opt.png"))
        plt.ioff()
        plt.close()
        return self.f, self.MSLL, np.array(trace_)

    def plot(self, fig, f_, MSLL_, MSLL_f, indicator_):
        MSLL_ = MSLL_[np.argsort(MSLL_[:, :2])]
        MSLL_f = MSLL_f[np.argsort(MSLL_f[:, :2])]
        points = self.site[np.isin(self.site[:, 0], f_)]
        plt.clf()
        ST = cal_time(time.time() - self.initial_time)
        fig.suptitle("Iter: {} \nTotally spend: {}h{}min{}secs"
                     .format(self.gen, int(ST[0]), int(ST[1]), int(ST[2])), fontweight="bold")

        grid = plt.GridSpec(3, 7, wspace=0.5, hspace=0.5)
        plt.subplot(grid[:, :2])
        plt.scatter(self.pop[:, 1], self.pop[:, 2], color="green", label="Demand points")
        plt.scatter(points[:, 1], points[:, 2], color="red", label="Target sites")
        plt.title("Location of optimal sites", fontweight="bold")
        plt.legend(loc="best")
        plt.axis("off")

        ax1 = plt.subplot(grid[:1, 2:7])
        ax1.plot([i + 1 for i in range(len(indicator_))], indicator_[:, 0, 0],
                 color="lightcoral", label="Average obj-1")
        ax1.plot([i + 1 for i in range(len(indicator_))], indicator_[:, 1, 0],
                 color="darkred", label="Optimal obj-1")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Obj-1 val: accessibility")
        ax1.legend(loc="best")
        plt.title("Obj-1: Maximize the accessibility", fontweight="bold")

        ax2 = plt.subplot(grid[1:2, 2:7])
        ax2.plot([i + 1 for i in range(len(indicator_))], 1/indicator_[:, 0, 1],
                 color="lightblue", label="Average obj-2")
        ax2.plot([i + 1 for i in range(len(indicator_))], 1/indicator_[:, 1, 1],
                 color="darkblue", label="Optimal obj-2")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Obj-2 val: distance")
        ax2.legend(loc="best")
        plt.title("Obj-2: Minimize the distance", fontweight="bold")

        ax3 = plt.subplot(grid[2:, 2:7])
        ax3.plot([i + 1 for i in range(len(indicator_))], 1 / indicator_[:, 0, 2],
                 color="lightgreen", label="Average obj-3")
        ax3.plot([i + 1 for i in range(len(indicator_))], 1 / indicator_[:, 1, 2],
                 color="darkgreen", label="Optimal obj-3")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Obj-3 val: drawback")
        ax3.legend(loc="best")
        plt.title("Obj-3: Minimize the drawback", fontweight="bold")

    def out_para(self):
        ST = cal_time(time.time() - self.initial_time)
        para_series = pd.Series({"NS": self.NS, "NP": self.NP, "Ncl": self.Ncl, "max_PF": self.max_PF,  "pm": self.pm,
                                 "Spend_time": "{}h{}min{}secs".format(int(ST[0]), int(ST[1]), int(ST[2]))},
                                name="Value")
        return para_series


class OutData:
    def __init__(self, filepath, site, f_, MSLL_, trace_, param_, shp_crs=None):
        self.filepath = create_dir(
            os.path.join(filepath, "Multi_Opti_{}".format(time.strftime("%Y-%m-%d %H-%M-%S"))))
        self.site = site
        args = np.argsort(MSLL_[:, 2])
        self.f_ = [f_[i] for i in args]
        self.MSLL_ = MSLL_[args]
        self.trace_ = trace_
        self.param_ = param_
        if shp_crs is not None:
            self.crs = shp_crs
            self.__out_shp()
        self.__out_img()
        self.__out_TAB()

    def __out_shp(self):
        shp_path = create_dir(os.path.join(self.filepath, "SHP"))
        for i in range(len(self.f_)):
            arr_ = self.site[np.isin(self.site[:, 0], self.f_[i])]
            self.__create_shp(os.path.join(shp_path, "Result_{}.shp".format(i + 1)), arr_)

    def __out_img(self):
        img_path = create_dir(os.path.join(self.filepath, "IMG"))

        grid = plt.GridSpec(3, 6, wspace=0.5, hspace=0.5)
        ax1 = plt.subplot(grid[:1, :])
        ax1.plot([i + 1 for i in range(len(self.trace_))], self.trace_[:, 0, 0], color="lightcoral",
                 label="Average obj-1")
        ax1.plot([i + 1 for i in range(len(self.trace_))], self.trace_[:, 1, 0], color="darkred", label="Optimal obj-1")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Obj-1 val: accessibility")
        ax1.legend(loc="best")
        plt.title("Obj-1: Maximize the accessibility", fontweight="bold")

        ax2 = plt.subplot(grid[1:2, :])
        ax2.plot([i + 1 for i in range(len(self.trace_))], self.trace_[:, 0, 1], color="lightblue",
                 label="Average obj-2")
        ax2.plot([i + 1 for i in range(len(self.trace_))], self.trace_[:, 1, 1], color="darkblue",
                 label="Optimal obj-2")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Obj-2 val: distance")
        ax2.legend(loc="best")
        plt.title("Obj-2: Minimize the distance", fontweight="bold")
        plt.savefig(os.path.join(img_path, "ObjValues.png"))
        plt.close()

        ax3 = plt.subplot(grid[2:3, :])
        ax3.plot([i + 1 for i in range(len(self.trace_))], 1 / self.trace_[:, 0, 2], color="lightgreen",
                 label="Average obj-3")
        ax3.plot([i + 1 for i in range(len(self.trace_))], 1 / self.trace_[:, 1, 2], color="darkgreen",
                 label="Optimal obj-3")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Obj-3 val: drawback")
        ax3.legend(loc="best")
        plt.title("Obj-3: Minimize the drawback", fontweight="bold")
        plt.savefig(os.path.join(img_path, "ObjValues.png"))
        plt.close()

    def __out_TAB(self):

        tab_path = create_dir(os.path.join(self.filepath, "TAB"))
        self.param_.to_excel(os.path.join(tab_path, "Parameters.xlsx"))
        df_ = pd.DataFrame({"ID": [i + 1 for i in range(len(self.f_))],
                            "Obj-1": self.MSLL_[:, 0],
                            "Obj-2": 1 / self.MSLL_[:, 1],
                            "Obj-3": self.MSLL_[:, 2]})
        df_.to_excel(os.path.join(tab_path, "ObjValues.xlsx"))



    def __create_shp(self, filepath, arr_):
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(filepath)
        layer = data_source.CreateLayer("Result", self.crs, ogr.wkbPoint)

        field_name = ogr.FieldDefn("FID", ogr.OFTString)
        layer.CreateField(field_name)
        for row in arr_:
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("FID", int(row[0]))
            wkt = "POINT(%f %f %f)" % (float(row[1]), float(row[2]), float(row[3]))
            point = ogr.CreateGeometryFromWkt(wkt)
            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feature = None
        data_source = None


class GetInput:
    def __init__(self, P1_path, P2_path, OD_path, fields):
        """
        :param P1_path: P1文件路径
        :param OD_path: OD文件路径
        :param fields: OD所需字段名称，如fields=["P1_ID", "P2_ID", "Distance", "Population", "ai1", "score"]

        """
        print("Loading P1...")
        self.P1, self.crs, = self.get_shp(P1_path)
        print("Loading P2...")
        self.P2, _ = self.get_shp(P2_path)
        print("Loading OD...")
        self.OD = pd.read_csv(OD_path)
        self.OD = self.OD[fields].values



    def standard_input(self):
        return self.P1, self.P2, self.OD, self.crs

    @staticmethod
    def get_shp(shp_path):
        """
        :param shp_path: 矢量文件路径
        :return: arr_——由 [P1_ID, x, y] 组成的 n*3 数组， crs_——投影坐标数据
        """
        shp_driver = ogr.GetDriverByName('ESRI Shapefile')
        shp_name = shp_path.split('\\')[-1]
        shp_source = shp_driver.Open(shp_path, 0)
        if shp_source is None:
            print("Failed to load [{}]!".format(shp_path))
        shp_ = shp_source.GetLayerByIndex(0)
        print("\r[{}] is already loaded!".format(shp_name), end="")

        crs_ = shp_.GetSpatialRef()
        field_name = shp_.GetLayerDefn().GetFieldDefn(0).GetName()  # 获取id字段名称
        feature_num = shp_.GetFeatureCount()
        arr_ = np.zeros((feature_num, 3))
        for i, feature_ in enumerate(shp_):
            arr_[i][0] = feature_.GetField(field_name)  # id
            x_, y_ = feature_.GetGeometryRef().GetPoint(0)[:2]
            arr_[i][1], arr_[i][2] = x_, y_  # x, y
            print("\r[{}/{}]".format(i + 1, feature_num), end="")
        print("")
        shp_ = None
        return arr_, crs_


class Main:
    def __init__(self):
        out_data_path = r"E:\Eccho\nanjing\TestData0703\work"

        P1_path = r"E:\Eccho\nanjing\TestData0703\shp_zc\SelectPoint.shp"  # 供给点shp, id字段-Id（Id按FID顺序从1开始）
        P2_path = r"E:\Eccho\nanjing\TestData0703\shp_zc\DemandPoint.shp"  # 需求点shp, id字段-Id（Id按FID顺序从1开始）
        OD_path = r"E:\Eccho\nanjing\TestData0703\OD_1.csv"  # 路径距离求解结果
        P1, P2, OD, crs = GetInput(P1_path=P1_path, P2_path=P2_path, OD_path=OD_path,
                                   fields=["P1_ID", "P2_ID", "Distance", "Population", "Dis_hosp"]).standard_input()
        obj_func = ObjFunc(OD_array=OD, core=7)
        aia = IASS(func=obj_func, site=P1, pop=P2, NS=20, NP=100, Ncl=5, max_PF=50, Gen=2, pm=0.7)
        f, MSLL, trace = aia.main(out_data_path)
        param = aia.out_para()
        OutData(filepath=out_data_path, site=P1, f_=f, MSLL_=MSLL,
                trace_=trace, param_=param, shp_crs=crs)


if __name__ == "__main__":
    start_time = time.time()
    Main()  # 运行
    st = cal_time(time.time() - start_time)
    print("\nTotally spend {}:{}:{}".format(int(st[0]), int(st[1]), int(st[2])))
