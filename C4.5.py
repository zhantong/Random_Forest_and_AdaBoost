import math
import random
class Node():
	
	def __init__(self,locations,parent,features_usable,depth):
		self.feature_index=None
		self.features_usable=features_usable
		self.split_condition=None
		self.parent=parent
		self.children=None
		self.is_leaf=False
		self.depth=depth
		self.label=None
		self.locations=locations
		#print('featurs usable:',features_usable)
		#print(self.locations)
		#print([Node.data['rows'][index] for index in self.locations])
		#print([Node.data['labels']['data'][index] for index in self.locations])
		#print([Node.data['labels']['data'][index] for index in [55,79]])
		if len(locations)==0:
			self.is_leaf=True
			self.label=self.parent.dominate_label()
			#print('leaf: locations=0',self.depth,self.locations)
		elif len(set([Node.data['labels'][index] for index in self.locations]))==1:
			self.is_leaf=True
			self.label=Node.data['labels'][self.locations[0]]
			#print('leaf: belong to 1 label',self.depth,self.label)
		elif (not self.features_usable) or self.all_same():
			self.is_leaf=True
			self.label=self.dominate_label()
			#print('leaf: no feature usable',self.depth,self.locations)
	def all_same(self):
		for index in self.locations[1:]:
			for feature_index in self.features_usable:
				if Node.data['colomns'][feature_index]['data'][index]!=Node.data['colomns'][feature_index]['data'][self.locations[0]]:
					return False
		return True
	def dominate_label(self):
		lists=[Node.data['labels'][index] for index in self.locations]
		s=list(set(lists))
		m=-1
		index=-1
		for label in s:
			c=lists.count(label)
			if c>m:
				m=c
				index=label
		#print(index,m)
		return index

		#print(s)
		#print(max(enumerate([lists.count(x) for x in s]),key=lambda p:p[1]))
		#max_index,max_count=max(enumerate([lists.count(x) for x in s]),key=lambda p:p[1])
		#return s[max_index]
	def test(self):
		print(self.depth,self.locations,[Node.data['labels'][index] for index in self.locations])
		if self.is_leaf:
			print('leaf',self.label)
			return
		for child in self.children:
			child.test()
	def info_gain_ratio(self):
		result=[]
		label_entropy=entropy(Node.data['labels'],self.locations)
		info_gains=[]
		split_infos=[]
		for feature_index in self.features_usable:
			if Node.data['colomns'][feature_index]['is_discrete']:
				colomn=Node.data['colomns'][feature_index]['data']
				info_e=info_entropy(colomn,Node.data['labels'],self.locations)
				info_gain=label_entropy-info_e
				split_info=entropy(colomn,self.locations)
			else:
				info_gain,split_info=numeric_info_entropy(label_entropy,Node.data['colomns'][feature_index],Node.data['labels'],self.locations)
			info_gains.append(info_gain)
			split_infos.append(split_info)
			#print(info_e,info_gains)
		avg_split_info=sum(split_infos)/len(split_infos)
		#print(self.features_usable,self.locations)
		for info_gain,split_info in zip(info_gains,split_infos):
			result.append(info_gain/(split_info+avg_split_info))
		return result
	def index_max_info_gain_ratio(self):
		lists=self.info_gain_ratio()
		return self.features_usable[lists.index(max(lists))]
	def split_to(self,sample_index):
		value=Node.data['colomns'][self.feature_index]['data'][sample_index]
		return self.split_condition.index(value)
	def do(self):
		
		if self.is_leaf:
			return
		self.feature_index=self.index_max_info_gain_ratio()
		features_usable=self.features_usable[:]
		features_usable.remove(self.feature_index)
		feature=Node.data['colomns'][self.feature_index]['data']
		

		if Node.data['colomns'][self.feature_index]['is_discrete']:
			
			s=list(set(feature))
			condition_num=len(s)
			self.split_condition=s
			locations_list=[[] for x in s]
			for index in self.locations:
				locations_list[s.index(feature[index])].append(index)
			
		else:
			condition_num=2
			self.split_condition=Node.data['colomns'][self.feature_index]['threshold']
			locations_list=[[],[]]
			for index in self.locations:
				if feature[index]<=self.split_condition:
					locations_list[0].append(index)
				else:
					locations_list[1].append(index)
		#print('inner:',self.depth,self.feature_index,self.split_condition,self.locations,self.features_usable)
		self.children=[None]*condition_num
		for i in range(condition_num):
			self.children[i]=Node(locations_list[i],self,features_usable,self.depth+1)
			self.children[i].do()
	def get(self,sample):
		#print(self.depth)
		if self.is_leaf:
			return self.label
		if Node.data['colomns'][self.feature_index]['is_discrete']:
			split_index=self.split_condition.index(sample[self.feature_index])
		else:
			if sample[self.feature_index]<=self.split_condition:
				split_index=0
			else:
				split_index=1
		return self.children[split_index].get(sample)
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
		#print(i,length,count1,count2,t1,t2)
		if t1!=0 and t1!=1:
			ie+=d1*(-t1*math.log2(t1)-(1-t1)*math.log2(1-t1))
		if t2!=0 and t2!=1:
			ie+=(1-d1)*(-t2*math.log2(t2)-(1-t2)*math.log2(1-t2))
		ig=label_entropy-ie
		si=-d1*math.log2(d1)-(1-d1)*math.log2(1-d1)
		igr=ig/si
		igrs.append((ig,si,i-1,igr))
	m=max(igrs,key=lambda x:x[3])
	colomn['threshold']=data[m[2]]
	return m[0],m[1]

class C_4_point_5():
	def __init__(self,file_path):
		self.data={
		'rows':None,
		'colomns':None,
		'labels':None,
		}
		self.data['rows'],self.data['colomns'],self.data['labels']=self.read_from_file(file_path)
		pass
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
	def do(self):
		Node.data=self.data
		root=Node([x for x in range(len(self.data['labels']))],None,[x for x in range(len(self.data['colomns']))],0)
		root.do()
		#root.test()
		#print(root.get([4,1,3,0,1,0,1,4,1]))
		count=0
		for line,real_label in  zip(Node.data['rows'],Node.data['labels']):
			label=root.get(line)
			if label!=real_label:
				count+=1
		print(count)
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
		pass
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
	def do(self):
		Node.data=self.data
		root=Node([x for x in range(len(self.data['labels']))],None,[x for x in range(len(self.data['colomns']))],0)
		root.do()
		#root.test()
		#print(root.get([4,1,3,0,1,0,1,4,1]))
		count=0
		for line,real_label in  zip(Node.data['rows'],Node.data['labels']):
			label=root.get(line)
			if label!=real_label:
				count+=1
		print(count)
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
		self.T=100
		self.data=data
		self.locations=locations
		self.locations_num=len(self.locations)
		self.update=[]
		
	def do(self):
		t=0
		weight=[1/self.locations_num]*self.locations_num
		locations=self.locations

		features_usable=[x for x in range(len(self.data['colomns']))]
		while t<self.T:
			root=Node(locations,None,features_usable,0)
			root.do()
			validate=[]
			for index in locations:
				label=root.get(Node.data['rows'][index])
				if label==Node.data['labels'][index]:
					validate.append(True)
				else:
					validate.append(False)
			error_num=validate.count(False)
			update=math.log(validate.count(True)/error_num)/2
			self.update.append(update)
			for index,is_true in zip(locations,validate):
				if is_true:
					weight[index]*=math.exp(-update)
				else:
					weight[index]*=math.exp(update)
			sum_weight=sum(weight)
			weight=[w/sum_weight for w in weight]
			t+=1
			if error_num==0:
				break
	
if __name__=='__main__':
	#c=C_4_point_5('data/breast-cancer-assignment5.txt')
	#c.do()
	t=Ten_fold_cross_validation('data/breast-cancer-assignment5.txt')
	t.do_random_forest()