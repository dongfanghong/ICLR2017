import numpy as np
import scipy.io
import pickle
import sklearn.linear_model
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Pool

'''
fin = open('/afs/cs.stanford.edu/u/dfh13/list_attr_celeba.txt')
fin.readline()
fin.readline()

data = []
for line in fin.readlines():
	#print line
	row = line.strip('\n').strip('\r').replace('  ',' ').split(' ')[1:]
	#print row
	assert(len(row)==40)
	data.append(row)
	if len(data)%10000==0:
		print len(data)

attribute = np.array(data,dtype='int32')
np.save('attribute',attribute)
'''
attribute = np.load('/atlas/u/dfh13/attribute.npy')

print 'Loading features'

psize = 0
print psize

#feat = np.load('/atlas/u/dfh13/feat_tot_'+str(psize)+'/feat_database_whole.npy',mmap_mode='r')
feat = np.load('/atlas/u/dfh13/faces_feat_baseline_whole_database.npy',mmap_mode='r')
#feat = pickle.load(open('/atlas/u/dfh13/faces_feat_baseline_database.pkl','rb'))
#feat = np.array(feat)
#np.save('/atlas/u/dfh13/faces_feat_baseline_database',feat)

num_samples = feat.shape[0]
feat = feat.reshape((num_samples,-1))
print feat.shape
#scipy.io.savemat('/atlas/u/dfh13/attribute.mat',dict(attribute=array))

print 'Loading finished'

trainX = feat[:30000,:]
trainY = attribute[:30000,:]
print trainX.shape

testX = feat[100000:180000,:]
testY = attribute[100000:180000,:]
result = []

def work(i):
	import time
	#model = sklearn.linear_model.LogisticRegression()
	start = time.time()
	model = SVC(class_weight='balanced',C=0.5)
	#model = GradientBoostingClassifier()
	model.fit(trainX,trainY[:,i])
	pred = model.predict(testX)
	accuracy_pos = float(((pred==1)*(testY[:,i]==1)).sum())/(testY[:,i]==1).sum()
	accuracy_neg = float(((pred==-1)*(testY[:,i]==-1)).sum())/(testY[:,i]==-1).sum()
	accuracy = (accuracy_pos+accuracy_neg)/2

	pred = model.predict(testX)
	accuracy_pos = float(((pred==1)*(testY[:,i]==1)).sum())/(testY[:,i]==1).sum()
	accuracy_neg = float(((pred==-1)*(testY[:,i]==-1)).sum())/(testY[:,i]==-1).sum()
	accuracy = (accuracy_pos+accuracy_neg)/2

	pred_train = model.predict(trainX)
	accuracy_pos_train = float(((pred_train==1)*(trainY[:,i]==1)).sum())/(trainY[:,i]==1).sum()
	accuracy_neg_train = float(((pred_train==-1)*(trainY[:,i]==-1)).sum())/(trainY[:,i]==-1).sum()
	accuracy_train = (accuracy_pos_train+accuracy_neg_train)/2

	print i,accuracy,accuracy_pos,accuracy_neg,accuracy_train,accuracy_pos_train,accuracy_neg_train,time.time()-start
	return i,accuracy

pool = Pool(40)
result = pool.map(work,range(40))
print result

#fout = open('result_'+str(psize)+'.txt','w')
fout = open('log_plain.txt','w')
for entry in result:
	print >>fout, entry[1]
fout.close()