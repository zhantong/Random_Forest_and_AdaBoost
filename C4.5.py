import math
class Node():
	
	def __init__(self,locations,parent):
		self.feature_index=None
		self.split_condition=None
		self.parent=parent
		self.children=None
		self.is_leaf=False
		self.label=None
		self.locations=locations
		if len(locations)==0:
			self.is_leaf=True
			self.label=self.parent.dominate_label()
		if len(set([Node.data['labels']['data'][index] for index in self.locations]))==1:
			self.is_leaf=True
			self.label=Node.data['labels']['data'][0]
	def dominate_label(self):
		lists=[Node.data['labels']['data'][index] for index in self.locations]
		s=list(set(lists))
		max_index,max_count=max([lists.count(x) for x in s])
		return s[max_index]

	def info_gain_ratio(self):
		result=[]
		label_entropy=entropy(Node.data['labels']['data'],self.locations)
		for colomn in Node.data['colomns']:
			info_e=info_entropy(colomn['data'],Node.data['labels']['data'],self.locations)
			info_gain=label_entropy-info_e
			split_info=entropy(colomn['data'],self.locations)
			info_gain_ratio=info_gain/split_info
			result.append(info_gain_ratio)
		return result
	def index_max_info_gain_ratio(self):
		lists=self.info_gain_ratio()
		return lists.index(max(lists))
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
		locations_list=[[] for x in s]
		for index in self.locations:
			locations_list[s.index(feature[index])].append(index)
		self.children=[None]*condition_num
		for i in range(condition_num):
			self.children[i]=Node(locations_list[i],self)
			print('new node')
			self.children[i].do()


def entropy(data,locations):
	s=set([data[index] for index in locations])
	print(s)
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
		root=Node([x for x in range(len(self.data['labels']['data']))],None)
		root.do()
if __name__=='__main__':
	c=C_4_point_5('data/breast-cancer-assignment5.txt')
	c.do()