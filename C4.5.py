class Node():
	
	def __init__(self,data):
		self.feature=None
		self.threshold=None
		self.data=data
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
		Node.label=self.label
		root=Node(self.data)
if __name__=='__main__':
	c=C_4_point_5('data/breast-cancer-assignment5.txt')
	print(c.data)