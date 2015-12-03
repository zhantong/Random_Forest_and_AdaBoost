import math
import random
class Node():
	"""决策树节点类
	用来构建决策树
	"""
	def __init__(self,locations,parent,features_usable,depth):
		"""初始化
		必要的变量声明和赋值
		在赋值后立即判断是否为叶子节点
		"""
		self.feature_index=None
		self.features_usable=features_usable
		self.split_condition=None
		self.parent=parent
		self.children=None
		self.is_leaf=False
		self.depth=depth
		self.label=None
		self.locations=locations
		self.prun=0
		if len(locations)==0:#节点不含任何sample，则为叶子节点
			self.is_leaf=True
			self.label=self.parent.dominate_label()[0]#节点label为父节点sample中占多数的label
		elif len(set([Node.data['labels'][index] for index in self.locations]))==1:#所有sample的label相同，则为叶子节点
			self.is_leaf=True
			self.label=Node.data['labels'][self.locations[0]]#节点label为sample的label
		elif (not self.features_usable) or self.all_same():#没有可用的featue，或者所有sample值完全相同但label不同，则为叶子节点
			#关于所有sample值完全相同但label不同
			#有些sample数据除了label不同其他完全相同，这种是不可能继续分开的
			self.is_leaf=True
			self.label=self.dominate_label()[0]#节点label为此节点sample中占多数的label
	def all_same(self):
		"""此节点所有sample是否除label外完全相同
		为了解决多个sample除label外完全相同的问题
		"""
		for index in self.locations[1:]:
			for feature_index in self.features_usable:
				if Node.data['colomns'][feature_index]['data'][index]!=Node.data['colomns'][feature_index]['data'][self.locations[0]]:
					return False
		return True
	def dominate_label(self):
		"""此节点占多数的label
		统计此节点的sample中所有label，返回出现次数最多的
		"""
		lists=[Node.data['labels'][index] for index in self.locations]
		m,index=max(map(lambda x: (lists.count(x), x), lists))#出现次数，label值
		return index,m

	def recursion_test(self):
		"""递归测试函数
		深度优先遍历
		"""
		print(self.depth,self.locations,[Node.data['labels'][index] for index in self.locations])
		if self.is_leaf:
			print('leaf',self.label)#碰到叶子节点的操作
			return
		for child in self.children:
			child.test()
	def info_gain_ratio(self):
		"""计算所有可用feature的info gain ratio
		返回值为所有可用featrue的info gain ratio值
		用到avg_split_info是为了避免除法时分母为0的情况
		即避免此节点下某一feature的值全部相同
		"""
		result=[]#存储所有可用featrue的info gain ratio值
		label_entropy=entropy(Node.data['labels'],self.locations)
		info_gains=[]#存储所有可用featrue的info gain值
		split_infos=[]#存储所有可用featrue的split info值
		for feature_index in self.features_usable:
			if Node.data['colomns'][feature_index]['is_discrete']:#根据feature值为离散或连续分开处理
				colomn=Node.data['colomns'][feature_index]['data']
				info_e=info_entropy(colomn,Node.data['labels'],self.locations)#info entropy
				info_gain=label_entropy-info_e
				split_info=entropy(colomn,self.locations)
			else:#连续型数据由另外的函数计算最佳值后返回到本函数
				info_gain,split_info=numeric_info_entropy(label_entropy,Node.data['colomns'][feature_index],Node.data['labels'],self.locations)
			info_gains.append(info_gain)
			split_infos.append(split_info)
		avg_split_info=sum(split_infos)/len(split_infos)#避免split info为0
		for info_gain,split_info in zip(info_gains,split_infos):
			result.append(info_gain/(split_info+avg_split_info))#计算info gain ratio
		return result
	def index_max_info_gain_ratio(self):
		"""返回可用feature中info gain ratio最大的索引
		即从info_gain_ratio()中挑出最大值
		"""
		lists=self.info_gain_ratio()
		return self.features_usable[lists.index(max(lists))]
	def do(self):
		"""递归建树
		由根节点触发，递归建立决策树
		如果为内部节点则根据info gain ratio分裂产生子树
		"""
		if self.is_leaf:#如果为叶子节点，则递归完毕，返回时计算剪枝要用的参数
			self.prun=len(self.locations)-[Node.data['labels'][index] for index in self.locations].count(self.label)+0.5
			return
		self.feature_index=self.index_max_info_gain_ratio()#以哪个feature进行分裂
		features_usable=self.features_usable[:]
		features_usable.remove(self.feature_index)#传递给子节点的可用feature列表，将此节点用到的feature移除
		feature=Node.data['colomns'][self.feature_index]['data']#此分裂feature的值,list
		if Node.data['colomns'][self.feature_index]['is_discrete']:#如果此feature值为离散点
			s=list(set(feature))#feature值集合（去重）
			condition_num=len(s)#子节点个数
			self.split_condition=s#查找时根据此值选择子节点
			locations_list=[[] for x in s]#初始化所有子节点分配到的sample
			for index in self.locations:#对每个sample进行分配
				locations_list[s.index(feature[index])].append(index)#分配到对应的子节点
		else:#如果此feature值为连续值
			condition_num=2#连续值一定二分
			self.split_condition=Node.data['colomns'][self.feature_index]['threshold']#得到预先计算得到的划分点
			locations_list=[[],[]]#初始化两个子节点分配到的sample
			for index in self.locations:#对每个sample进行分配
				if feature[index]<=self.split_condition:#值小于划分点值则分到第一个子节点
					locations_list[0].append(index)
				else:#值大于划分点值则分到第二个子节点
					locations_list[1].append(index)
		self.children=[None]*condition_num#初始化子节点list
		for i in range(condition_num):
			self.children[i]=Node(locations_list[i],self,features_usable,self.depth+1)#对子节点分类sample，可用的feature
			self.children[i].do()#子节点执行递归
			self.prun+=self.children[i].prun#子节点返回时更新剪枝所需参数
	def get(self,sample):
		"""输入sample，输出决策树结果label
		此方法应当只用于根节点
		即决策树的查找方法
		"""
		if self.is_leaf:#递归碰到叶子节点则返回
			return self.label
		if Node.data['colomns'][self.feature_index]['is_discrete']:#离散值则根据匹配情况选择子节点
			split_index=self.split_condition.index(sample[self.feature_index])
		else:#连续值则与划分点比较，选择子节点
			if sample[self.feature_index]<=self.split_condition:
				split_index=0
			else:
				split_index=1
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
		if self.is_leaf:#不处理叶子节点
			return
		sample_num=len(self.locations)
		error_num=sample_num-self.dominate_label()[1]+0.5
		leaf_num=self.prun
		#
		if leaf_num>sample_num or error_num<=leaf_num+math.sqrt(leaf_num*(1-leaf_num/sample_num)):
			self.is_leaf=True
			self.label=self.dominate_label()[0]
			self.children=None
			return
		for child in self.children:
			child.pruning()
def entropy(data,locations):
	s=set([data[index] for index in locations])
	if len(s)==1:
		return 0
	count={}
	for dif in s:
		count[dif]=0
	
	for index in locations:
		count[data[index]]+=1
	length=len(locations)
	result=0
	for c in count.values():
		p=c/length
		result-=p*math.log2(p)
	return result
def info_entropy(feature,labels,locations):
	f_s=set([feature[index] for index in locations])
	if len(f_s)==1:
		return 0
	f_count={}
	for dif in f_s:
		f_count[dif]={
		'count':0,
		'locations':[]
		}
	for index in locations:
		f_count[feature[index]]['count']+=1
		f_count[feature[index]]['locations'].append(index)
	f_length=len(locations)
	result=0
	for c in f_count.values():
		result+=c['count']/f_length*entropy(labels,c['locations'])
	return result
def numeric_info_entropy(label_entropy,colomn,labels,locations):
	temp=sorted([(colomn['data'][index],labels[index]) for index in locations],key=lambda x:x[0])
	data=[x[0] for x in temp]
	label=[x[1] for x in temp]
	length=len(data)
	label_set=list(set(label))
	lab=label_set[0]
	count1=label.count(lab)
	igrs=[]
	for i in range(1,length):
		count2=label[0:i].count(lab)
		t1=count2/i
		t2=(count1-count2)/(length-i)
		d1=i/length
		ie=0
		if t1!=0 and t1!=1:
			ie+=d1*(-t1*math.log2(t1)-(1-t1)*math.log2(1-t1))
		if t2!=0 and t2!=1:
			ie+=(1-d1)*(-t2*math.log2(t2)-(1-t2)*math.log2(1-t2))
		ig=label_entropy-ie
		si=-d1*math.log2(d1)-(1-d1)*math.log2(1-d1)
		igr=ig/si
		igrs.append((ig,si,i-1,igr))
	m=max(igrs,key=lambda x:x[0])
	colomn['threshold']=data[m[2]]
	return m[0],m[1]

class Ten_fold_cross_validation():
	def __init__(self,file_path):
		self.data={
		'rows':None,
		'colomns':None,
		'labels':None,
		}
		self.data['rows'],self.data['colomns'],self.data['labels']=self.read_from_file(file_path)
		self.ten_rand=self.generate_ten_random()
		Node.data=self.data
	def read_from_file(self,file_path): 
		rows=[]
		colomns=[]
		labels=[]
		with open(file_path,'r') as f:
			for is_discrete in f.readline().strip().split(','):
				colomns.append({
					'is_discrete':is_discrete=='1',
					'data':[]
					})
			for line in f:
				row=[]
				for colomn,d in zip(colomns,line.strip().split(',')[:-1]):
					if colomn['is_discrete']:
						t=int(float(d))
						colomn['data'].append(t)
						row.append(t)
					else:
						t=float(d)
						colomn['data'].append(t)
						row.append(t)
				rows.append(row)
				labels.append(int(float(line.rsplit(',',1)[-1].strip())))
		for colomn in colomns:
			if colomn['is_discrete']:
				colomn['set']=set(colomn['data'])
		return rows,colomns,labels
	def generate_ten_random(self):
		ten=10
		length=len(self.data['labels'])
		rand_list=random.sample(range(length),length)
		ten_rand=[]
		step=int(length/ten)
		for i in range(0,length,step):
			ten_rand.append((rand_list[0:i]+rand_list[i+step:],rand_list[i:i+step]))
		return ten_rand
	def test(self):
		ten=10
		Node.data=self.data
		for i in range(ten):
			root=Node(self.ten_rand[i][0],None,[x for x in range(len(self.data['colomns']))],0)
			root.do()
			count=0
			for index in self.ten_rand[i][1]:
				label=root.get(Node.data['rows'][index])
				if label!=Node.data['labels'][index]:
					count+=1
			print(count,'/',len(self.ten_rand[i][1]))
	def do_random_forest(self):
		ten=10
		for i in range(ten):
			rf=Random_forest(self.data,self.ten_rand[i][0])
			rf.build_forest()
			count=0
			for index in self.ten_rand[i][1]:
				label=rf.validate(Node.data['rows'][index])
				if label!=Node.data['labels'][index]:
					count+=1
			print(count,'/',len(self.ten_rand[i][1]))
	def do_adaboost(self):
		ten=10
		for i in range(ten):
			rf=Adaboost(self.data,self.ten_rand[i][0])
			rf.do()
			count=0
			for index in self.ten_rand[i][1]:
				label=rf.validate(Node.data['rows'][index])
				if label!=Node.data['labels'][index]:
					count+=1
			print(count,'/',len(self.ten_rand[i][1]))
class Random_forest():
	def __init__(self,data,locations):
		self.tree_num=30
		self.data=data
		self.feature_num=len(data['colomns'])
		self.feature_rand_num=round(math.log2(self.feature_num))+1
		self.locations=locations
		self.locations_num=len(self.locations)
		self.trees=[]
	def build_forest(self):
		for i in range(self.tree_num):
			locations=[random.choice(self.locations) for x in range(self.locations_num)]
			features_usable=random.sample(range(self.feature_num),self.feature_rand_num)
			root=Node(locations,None,features_usable,0)
			root.do()
			self.trees.append(root)
	def validate(self,sample):
		result=[]
		for i in range(self.tree_num):
			label=self.trees[i].get(sample)
			result.append(label)
		return max(map(lambda x: (result.count(x), x), result))[1]

class Adaboost():
	def __init__(self,data,locations):
		self.T=10
		self.data=data
		self.locations=locations
		self.locations_num=len(self.locations)
		self.update=[]
		self.trees=[]
		self.label_set=sorted(list(set(Node.data['labels'])))
		self.label_middle=sum(self.label_set)/2
	def do(self):
		t=0
		ZOOM=self.locations_num*100
		weight=[1/self.locations_num]*self.locations_num
		locations=self.locations

		features_usable=[x for x in range(len(self.data['colomns']))]
		while t<self.T:
			root=Node(locations,None,features_usable,0)
			root.do()
			root.pruning()
			self.trees.append(root)
			validate=[]
			for index in self.locations:
				label=root.get(Node.data['rows'][index])
				if label==Node.data['labels'][index]:
					validate.append(True)
				else:
					validate.append(False)
			error_num=validate.count(False)
			if error_num==0:
				break
			update=math.log(validate.count(True)/error_num)/2
			self.update.append(update)
			for location,is_true in zip(locations,validate):
				index=self.locations.index(location)
				if is_true:
					weight[index]*=math.exp(-update)
				else:
					weight[index]*=math.exp(update)
			sum_weight=sum(weight)
			weight=[w/sum_weight for w in weight]
			rand=[]
			for location,w in zip(self.locations,weight):
				rand+=[location]*int(w*ZOOM)
			locations=random.sample(rand,self.locations_num)
			t+=1
	def validate(self,sample):
		result=0
		for tree,w in zip(self.trees,self.update):
			label=tree.get(sample)
			result+=label*w
			result/=len(self.trees)
		if result<=self.label_middle:
			return self.label_set[0]
		else:
			return self.label_set[1]

if __name__=='__main__':
	t=Ten_fold_cross_validation('data/german-assignment5.txt')
	#t.do_random_forest()
	print('-'*20)
	t.do_adaboost()