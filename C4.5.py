class C_4_point_5():
	def __init__(self):
		self.data=[]
		pass
	def read_from_file(self,file_path):
		data=[]
		with open(file_path,'r') as f:
			for is_discrete in f.readline().strip().split(','):
				print(is_discrete)
				data.append({
					'type':is_discrete=='1',
					'data':[]
					})
			for line in f:
				for feature,d in zip(data,line.strip().split(',')):
					feature['data'].append(d)
		return data

if __name__=='__main__':
	c=C_4_point_5()
	d=c.read_from_file('data/breast-cancer-assignment5.txt')
	print(d)