import math
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
		elif len(set([Node.data['labels']['data'][index] for index in self.locations]))==1:
			self.is_leaf=True
			self.label=Node.data['labels']['data'][self.locations[0]]
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
		lists=[Node.data['labels']['data'][index] for index in self.locations]
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
		print(self.depth,self.locations,[Node.data['labels']['data'][index] for index in self.locations])
		if self.is_leaf:
			print('leaf',self.label)
			return
		for child in self.children:
			child.test()
	def info_gain_ratio(self):
		result=[]
		label_entropy=entropy(Node.data['labels']['data'],self.locations)
		info_gains=[]
		split_infos=[]
		for feature_index in self.features_usable:
			colomn=Node.data['colomns'][feature_index]['data']
			info_e=info_entropy(colomn,Node.data['labels']['data'],self.locations)
			info_gains.append(label_entropy-info_e)
			split_infos.append(entropy(colomn,self.locations))
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
		feature=Node.data['colomns'][self.feature_index]['data']
		s=list(set(feature))
		condition_num=len(s)
		self.split_condition=s
		features_usable=self.features_usable[:]
		features_usable.remove(self.feature_index)
		locations_list=[[] for x in s]
		for index in self.locations:
			locations_list[s.index(feature[index])].append(index)
		self.children=[None]*condition_num
		#print('inner:',self.depth,self.feature_index,self.split_condition,self.locations,self.features_usable)
		for i in range(condition_num):
			self.children[i]=Node(locations_list[i],self,features_usable,self.depth+1)
			self.children[i].do()
	def get(self,sample):
		#print(self.depth)
		if self.is_leaf:
			return self.label
		split_index=self.split_condition.index(sample[self.feature_index])
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
		labels={
		'data':[],
		'set':None
		}
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
				labels['data'].append(int(float(line.rsplit(',',1)[-1].strip())))
		for colomn in colomns:
			if colomn['is_discrete']:
				colomn['set']=set(colomn['data'])
		labels['set']=set(labels['data'])
		return rows,colomns,labels
	def do(self):
		Node.data=self.data
		root=Node([x for x in range(len(self.data['labels']['data']))],None,[x for x in range(len(self.data['colomns']))],0)
		root.do()
		#root.test()
		#print(root.get([4,1,3,0,1,0,1,4,1]))
		for line,real_label in  zip(Node.data['rows'],Node.data['labels']['data']):
			label=root.get(line)
			print(label,real_label)
if __name__=='__main__':
	c=C_4_point_5('data/breast-cancer-assignment5.txt')
	c.do()