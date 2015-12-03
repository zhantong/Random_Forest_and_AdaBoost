"""C4.5决策树, Random Forest, AdaBoost Algorithm, 10 fold corss validation
四个类全部共用一个总的数据结构，即
data={
	'colomns':{
	'is_discrete':True/False,
	'data':[colomn 1,colomn 2, ..., colomn m]
	},
	'rows':[row 1,row 2, ..., row n],
	'labels':[label 1, label 2, ..., label n]
}
colomn x=[data x1, data x2, ..., data xn]
row x=[data 1x, data 2x, ..., data mx]
其中m为feature个数，n为sample个数
其中data['rows']只用在对sample验证时

在整个计算过程中，绝大多数为传递索引(index)，即指定哪些sample能用，哪些feature能用
而尽量不构造新的数据，只传递索引
"""

import math
import random


class Node():

    """决策树节点类
    用来构建C4.5决策树
    决策树能够处理连续型数据
        具有剪枝功能
    """

    def __init__(self, locations, parent, features_usable, depth):
        """初始化
        必要的变量声明和赋值
        在赋值后立即判断是否为叶子节点
        """
        self.feature_index = None
        self.features_usable = features_usable
        self.split_condition = None
        self.parent = parent
        self.children = None
        self.is_leaf = False
        self.depth = depth
        self.label = None
        self.locations = locations
        self.prun = 0
        if len(locations) == 0:  # 节点不含任何sample，则为叶子节点
            self.is_leaf = True
            # 节点label为父节点sample中占多数的label
            self.label = self.parent.dominate_label()[0]
        # 所有sample的label相同，则为叶子节点
        elif len(set([Node.data['labels'][index] for index in self.locations])) == 1:
            self.is_leaf = True
            self.label = Node.data['labels'][
                self.locations[0]]  # 节点label为sample的label
        # 没有可用的featue，或者所有sample值完全相同但label不同，则为叶子节点
        elif (not self.features_usable) or self.all_same():
            # 关于所有sample值完全相同但label不同
            # 有些sample数据除了label不同其他完全相同，这种是不可能继续分开的
            self.is_leaf = True
            # 节点label为此节点sample中占多数的label
            self.label = self.dominate_label()[0]

    def all_same(self):
        """此节点所有sample是否除label外完全相同
        为了解决多个sample除label外完全相同的问题
        """
        for index in self.locations[1:]:
            for feature_index in self.features_usable:
                if Node.data['colomns'][feature_index]['data'][index] != Node.data['colomns'][feature_index]['data'][self.locations[0]]:
                    return False
        return True

    def dominate_label(self):
        """此节点占多数的label
        统计此节点的sample中所有label，返回出现次数最多的
        """
        lists = [Node.data['labels'][index] for index in self.locations]
        # 出现次数，label值
        m, index = max(map(lambda x: (lists.count(x), x), lists))
        return index, m

    def recursion_test(self):
        """递归测试函数
        深度优先遍历
        """
        print(self.depth, self.locations, [
              Node.data['labels'][index] for index in self.locations])
        if self.is_leaf:
            print('leaf', self.label)  # 碰到叶子节点的操作
            return
        for child in self.children:
            child.test()

    def info_gain_ratio(self):
        """计算所有可用feature的info gain ratio
        返回值为所有可用featrue的info gain ratio值
        用到avg_split_info是为了避免除法时分母为0的情况
        即避免此节点下某一feature的值全部相同
        """
        result = []  # 存储所有可用featrue的info gain ratio值
        label_entropy = entropy(Node.data['labels'], self.locations)
        info_gains = []  # 存储所有可用featrue的info gain值
        split_infos = []  # 存储所有可用featrue的split info值
        for feature_index in self.features_usable:
            # 根据feature值为离散或连续分开处理
            if Node.data['colomns'][feature_index]['is_discrete']:
                colomn = Node.data['colomns'][feature_index]['data']
                info_e = info_entropy(
                    colomn, Node.data['labels'], self.locations)  # info entropy
                info_gain = label_entropy-info_e
                split_info = entropy(colomn, self.locations)
            else:  # 连续型数据由另外的函数计算最佳值后返回到本函数
                info_gain, split_info = numeric_info_entropy(label_entropy, Node.data[
                                                             'colomns'][feature_index], Node.data['labels'], self.locations)
            info_gains.append(info_gain)
            split_infos.append(split_info)
        avg_split_info = sum(split_infos)/len(split_infos)  # 避免split info为0
        for info_gain, split_info in zip(info_gains, split_infos):
            # 计算info gain ratio
            result.append(info_gain/(split_info+avg_split_info))
        return result

    def index_max_info_gain_ratio(self):
        """返回可用feature中info gain ratio最大的索引
        即从info_gain_ratio()中挑出最大值
        """
        lists = self.info_gain_ratio()
        return self.features_usable[lists.index(max(lists))]

    def do(self):
        """递归建树
        由根节点触发，递归建立决策树
        如果为内部节点则根据info gain ratio分裂产生子树
        """
        if self.is_leaf:  # 如果为叶子节点，则递归完毕，返回时计算剪枝要用的参数
            self.prun = len(self.locations)-[Node.data['labels'][index]
                                             for index in self.locations].count(self.label)+0.5
            return
        self.feature_index = self.index_max_info_gain_ratio()  # 以哪个feature进行分裂
        features_usable = self.features_usable[:]
        # 传递给子节点的可用feature列表，将此节点用到的feature移除
        features_usable.remove(self.feature_index)
        feature = Node.data['colomns'][self.feature_index][
            'data']  # 此分裂feature的值,list
        # 如果此feature值为离散点
        if Node.data['colomns'][self.feature_index]['is_discrete']:
            s = list(set(feature))  # feature值集合（去重）
            condition_num = len(s)  # 子节点个数
            self.split_condition = s  # 查找时根据此值选择子节点
            locations_list = [[] for x in s]  # 初始化所有子节点分配到的sample
            for index in self.locations:  # 对每个sample进行分配
                locations_list[s.index(feature[index])].append(
                    index)  # 分配到对应的子节点
        else:  # 如果此feature值为连续值
            condition_num = 2  # 连续值一定二分
            self.split_condition = Node.data['colomns'][
                self.feature_index]['threshold']  # 得到预先计算得到的划分点
            locations_list = [[], []]  # 初始化两个子节点分配到的sample
            for index in self.locations:  # 对每个sample进行分配
                if feature[index] <= self.split_condition:  # 值小于划分点值则分到第一个子节点
                    locations_list[0].append(index)
                else:  # 值大于划分点值则分到第二个子节点
                    locations_list[1].append(index)
        self.children = [None]*condition_num  # 初始化子节点list
        for i in range(condition_num):
            # 对子节点分类sample，可用的feature
            self.children[i] = Node(
                locations_list[i], self, features_usable, self.depth+1)
            self.children[i].do()  # 子节点执行递归
            self.prun += self.children[i].prun  # 子节点返回时更新剪枝所需参数

    def get(self, sample):
        """输入sample，输出决策树结果label
        此方法应当只用于根节点
        即决策树的查找方法
        """
        if self.is_leaf:  # 递归碰到叶子节点则返回
            return self.label
        # 离散值则根据匹配情况选择子节点
        if Node.data['colomns'][self.feature_index]['is_discrete']:
            split_index = self.split_condition.index(
                sample[self.feature_index])
        else:  # 连续值则与划分点比较，选择子节点
            if sample[self.feature_index] <= self.split_condition:
                split_index = 0
            else:
                split_index = 1
        return self.children[split_index].get(sample)

    def pruning(self):
        """剪枝
        采用悲观剪枝法
        对每个内部节点，如果满足条件则将此节点设为叶子节点，label值为占大多数的label
        这里可能出现叶子节点错误总和数目比所有节点数目大的情况
        原因是节点分裂时出现了很多空节点，这种分裂只是增加了树的深度
        而对决策没有好处，也应该被剪枝掉
        悲观剪枝的条件：
        此节点错误数<=所有子节点错误数之和+标准误差
        注意这里对于每个叶子节点会有0.5的增加值，此节点错误数也会有0.5增加值		
        """
        if self.is_leaf:  # 不处理叶子节点
            return
        sample_num = len(self.locations)
        error_num = sample_num-self.dominate_label()[1]+0.5
        leaf_num = self.prun
        # 叶子节点错误总和数目>所有节点数目 或 此节点错误数<=所有子节点错误数之和+标准误差
        if leaf_num > sample_num or error_num <= leaf_num+math.sqrt(leaf_num*(1-leaf_num/sample_num)):
            self.is_leaf = True
            self.label = self.dominate_label()[0]  # 节点label设为占大多数的bael
            self.children = None  # 释放子节点
            return
        for child in self.children:  # 不满足剪枝条件则向下递归
            child.pruning()


def entropy(data, locations):
    """计算离散型数据的entropy
    data为一个feature的所有数据，或者label的所有数据
    locations为计算entropy的所有索引
    """
    s = set([data[index] for index in locations])  # 获取集合（去重）
    if len(s) == 1:  # 如果只有一种值，则直接返回0
        return 0
    count = {}
    for dif in s:  # 初始化
        count[dif] = 0

    for index in locations:  # 计算每种值出现次数
        count[data[index]] += 1
    length = len(locations)
    result = 0
    for c in count.values():  # 计算entropy
        p = c/length
        result -= p*math.log2(p)
    return result


def info_entropy(feature, labels, locations):
    """计算离散型数据的info entropy
    需要一个feature以及所有的label
    locations为取值的索引
    """
    f_s = set([feature[index] for index in locations])  # 此feature的数据集合（去重）
    if len(f_s) == 1:  # 如果只有一种数据则直接返回0
        return 0
    f_count = {}
    for dif in f_s:  # 初始化
        f_count[dif] = {
            'count': 0,
            'locations': []
        }
    for index in locations:  # 统计每种值出现次数，和位置索引
        f_count[feature[index]]['count'] += 1
        f_count[feature[index]]['locations'].append(index)
    f_length = len(locations)
    result = 0
    for c in f_count.values():  # 计算info entropy
        result += c['count']/f_length*entropy(labels, c['locations'])
    return result


def numeric_info_entropy(label_entropy, colomn, labels, locations):
    """计算连续型数据的info entropy, info gain, split info (和 info gain ratio)
    返回的是对应info gain最大的连续值划分
    因为连续型数据处理一定是二分，
    所以采用简单的循环计算每种划分的info gain
    取值最大的划分，返回info gain和split info
    """
    temp = sorted([(colomn['data'][index], labels[index])
                   for index in locations], key=lambda x: x[0])  # 连续型值排序
    data = [x[0] for x in temp]  # 有序连续型值
    label = [x[1] for x in temp]  # 对应的label值
    length = len(data)
    label_set = list(set(label))  # label种类
    lab = label_set[0]  # label只有两种，任取一个作标记
    count1 = label.count(lab)  # 标记的label出现次数
    igrs = []  # 保存每种划分的info gain, split info, 划分索引, info gain ratio
    for i in range(1, length):  # 遍历划分
        # 下面的计算与正常计算完全一样，但为了节省计算量，作了一些优化
        count2 = label[0:i].count(lab)  # 前半部分标记label的出现次数
        t1 = count2/i  # 其出现次数占前半部分的比例
        t2 = (count1-count2)/(length-i)  # 后半部分标记label占后办部分的比例
        d1 = i/length  # 前半部分划分长度占总长度比例
        ie = 0  # info entropy
        if t1 != 0 and t1 != 1:  # 出现极端情况，避免log(0)报错
            ie += d1*(-t1*math.log2(t1)-(1-t1)*math.log2(1-t1))
        if t2 != 0 and t2 != 1:  # 出现极端情况，避免log(0)报错
            ie += (1-d1)*(-t2*math.log2(t2)-(1-t2)*math.log2(1-t2))
        ig = label_entropy-ie  # info gain
        si = -d1*math.log2(d1)-(1-d1)*math.log2(1-d1)  # split info
        igr = ig/si  # info gain ratio
        igrs.append((ig, si, i-1, igr))  # 收集每种划分的结果
    m = max(igrs, key=lambda x: x[0])  # 选取info gain最大的情况
    colomn['threshold'] = data[m[2]]  # 将划分值存到此feature中（因为每种feature只会划分一次）
    return m[0], m[1]  # 返回info gain, split info，与离散型对应


class Ten_fold_cross_validation():

    """10 fold cross validation
    """

    def __init__(self, file_path):
        """初始化
        必要的变量声明
        读取并处理文件数据
        生成10种随机training集和validation集
        设置Node类公共变量
        """
        self.data = {
            'rows': None,
            'colomns': None,
            'labels': None,
        }
        self.data['rows'], self.data['colomns'], self.data[
            'labels'] = self.read_from_file(file_path)
        self.ten_rand = self.generate_ten_random()
        Node.data = self.data

    def read_from_file(self, file_path):
        """读取并处理文件数据
        将文件分别按行、按列处理
        对于按列，以feature为单位分
        对于按行，以sample为单位分
        label部分单独提出来
        """
        rows = []
        colomns = []
        labels = []
        with open(file_path, 'r') as f:
            # 处理第一行，得到feature数目，是否为离散型
            for is_discrete in f.readline().strip().split(','):
                colomns.append({
                    'is_discrete': is_discrete == '1',  # 标记是否为离散型
                    'data': []
                })
            for line in f:  # 每次以行读取，分别按列和按行分类
                row = []
                for colomn, d in zip(colomns, line.strip().split(',')[:-1]):
                    if colomn['is_discrete']:
                        t = int(float(d))
                        colomn['data'].append(t)
                        row.append(t)
                    else:
                        t = float(d)
                        colomn['data'].append(t)
                        row.append(t)
                rows.append(row)
                labels.append(int(float(line.rsplit(',', 1)[-1].strip())))
        for colomn in colomns:
            if colomn['is_discrete']:
                colomn['set'] = set(colomn['data'])
        return rows, colomns, labels

    def generate_ten_random(self):
        """生成10种随机training集和validation集
        随机将数据集分为10份
        顺序以1份为validation集，剩下9份为training集
        这样得到10类（每类2份）数据集
        这里只需要对索引进行随机处理即可，
        不需要操作真正的数据
        """
        ten = 10
        length = len(self.data['labels'])
        rand_list = random.sample(range(length), length)  # 将索引值打乱，即为随机
        ten_rand = []
        start=0
        #这里为了保证每个valitaion集尽可能均衡，没有采用简单均值划分
        for i in range(ten,0,-1):
        	stop=start+int((length-start)/i)
        	ten_rand.append((rand_list[0:start]+rand_list[stop:],rand_list[start:stop]))
        	start=stop
        return ten_rand

    def test(self):
        """测试
        """
        ten = 10
        Node.data = self.data
        for i in range(ten):
            root = Node(
                self.ten_rand[i][0], None, [x for x in range(len(self.data['colomns']))], 0)
            root.do()
            count = 0
            for index in self.ten_rand[i][1]:
                label = root.get(Node.data['rows'][index])
                if label != Node.data['labels'][index]:
                    count += 1
            print(count, '/', len(self.ten_rand[i][1]))

    def do_random_forest(self):
        """对random forest测试
        一共进行10次测试
        """
        ten = 10
        for i in range(ten):  # 每次测试相对独立
            # 实例化random forest，传递数据，和training集
            rf = Random_forest(self.data, self.ten_rand[i][0])
            rf.build_forest()  # 建立random forest
            count = 0
            for index in self.ten_rand[i][1]:  # 对validation集的sample逐个验证
                # sample测试得到的label
                label = rf.validate(Node.data['rows'][index])
                if label != Node.data['labels'][index]:  # 与真实值比较
                    count += 1
            print(count, '/', len(self.ten_rand[i][1]))

    def do_adaboost(self):
        """对AdaBoost测试
        一共进行10次测试
        """
        ten = 10
        for i in range(ten):  # 每次测试相对独立
            # 实例化adaBoost，传递数据，和training集
            rf = Adaboost(self.data, self.ten_rand[i][0])
            rf.do()  # 运行AbaBoost算法
            count = 0
            for index in self.ten_rand[i][1]:  # 对validation集的sample逐个验证
                # sample测试得到的label
                label = rf.validate(Node.data['rows'][index])
                if label != Node.data['labels'][index]:  # 与真实值比较
                    count += 1
            print(count, '/', len(self.ten_rand[i][1]))


class Random_forest():

    """Random Forest
    """

    def __init__(self, data, locations):
        """必要的变量声明和赋值
        """
        self.tree_num = 30  # 设定森林中树的个数
        self.data = data
        self.feature_num = len(data['colomns'])  # feature的种数
        self.feature_rand_num = round(
            math.log2(self.feature_num))+1  # 每次随机取的feature的种数
        self.locations = locations  # training数据索引
        self.locations_num = len(self.locations)  # training数据个数
        self.trees = []  # 存储森林所有树的根节点

    def build_forest(self):
        """建立森林
        两次随机：
        随机重复取sample，数目为locations_num
        随机取feature，数目为feature_rand_num
        """
        for i in range(self.tree_num):
            # 随机重复取sample
            locations = [random.choice(self.locations)
                         for x in range(self.locations_num)]
            features_usable = random.sample(
                range(self.feature_num), self.feature_rand_num)  # 随机取feature
            root = Node(locations, None, features_usable, 0)  # 以上述取到的索引建立根节点
            root.do()  # 建树，Random Forest不需要进行剪枝
            self.trees.append(root)  # 将此树根节点加入到森林

    def validate(self, sample):
        """对sample测试其label
        """
        result = []  # 存储每个树返回的label
        for i in range(self.tree_num):  # 遍历每个树
            label = self.trees[i].get(sample)  # 此树返回的label
            result.append(label)
        # 选取出现次数最多的label作为结果
        return max(map(lambda x: (result.count(x), x), result))[1]


class Adaboost():

    """AdaBoost
    """

    def __init__(self, data, locations):
        """必要的变量声明和赋值
        """
        self.T = 10  # 设定循环次数
        self.data = data
        self.locations = locations
        self.locations_num = len(self.locations)  # training数据个数
        self.update = []  # 每次循环得到的update值（错误率相关）
        self.trees = []  # 存储每次循环得到的树的根节点
        # training集label集合，由小到大排序，实际最多只有2个
        self.label_set = sorted(list(set(Node.data['labels'])))
        # 应对label值不是-1 +1的情况，如0 1，这是label中间值
        self.label_middle = sum(self.label_set)/2

    def do(self):
        """运行AbaBoost算法
        C4.5不能处理sample的权重问题
        变通的方法是，每次得到新的weight集后，
        对training集以weight为概率重新采样，
        这样效果与以weight处理类似。
        python没有以概率采样，
        变通为概率较大者出现次数多，再进行随机采样
        """
        t = 0  # 标记第几次循环
        ZOOM = self.locations_num*100  # 对采样方法，提升精度
        weight = [1/self.locations_num]*self.locations_num  # 初始weight
        locations = self.locations  # 最初的training集
        # 可用feature为所有feature
        features_usable = [x for x in range(len(self.data['colomns']))]
        while t < self.T:
            # 以locations建树，locations每次都会更新
            root = Node(locations, None, features_usable, 0)
            root.do()
            root.pruning()  # 对决策树剪枝
            self.trees.append(root)  # 保存根节点
            validate = []  # 保存此次测试结果
            for index in self.locations:  # 每次建树后，以training集进行验证
                label = root.get(Node.data['rows'][index])
                if label == Node.data['labels'][index]:
                    validate.append(True)
                else:
                    validate.append(False)
            error_num = validate.count(False)  # 分类错误个数
            if error_num == 0:  # 没出错则提前停止循环
                break
            update = math.log(validate.count(True)/error_num)/2  # update值
            self.update.append(update)  # 保存，此权重作为验证时的参考
            for location, is_true in zip(locations, validate):  # 更新weight
                index = self.locations.index(location)
                if is_true:  # 分类正确的权重减少
                    weight[index] *= math.exp(-update)
                else:  # 分类错误的权重增加
                    weight[index] *= math.exp(update)
            sum_weight = sum(weight)
            weight = [w/sum_weight for w in weight]  # 归一化
            rand = []
            for location, w in zip(self.locations, weight):  # 以weight重新采样
                rand += [location]*int(w*ZOOM)
            random.shuffle(rand)
            locations = random.sample(rand, self.locations_num)
            t += 1

    def validate(self, sample):
        """对sample测试其label
        """
        result = 0
        # 对每棵树测试sample，得到label后乘以update值，累加后判断其最终label值
        for tree, w in zip(self.trees, self.update):
            label = tree.get(sample)
            result += label*w
            result /= len(self.trees)  # 这里取平均数避免label值非-1 +1时出错，如0 1
        if result <= self.label_middle:  # 根据label middle值判断其最终label值
            return self.label_set[0]
        else:
            return self.label_set[1]

if __name__ == '__main__':
    t = Ten_fold_cross_validation('data/german-assignment5.txt')
    # t.do_random_forest()
    print('-'*20)
    t.do_adaboost()
