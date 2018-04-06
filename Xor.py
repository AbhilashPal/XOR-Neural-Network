import numpy as np

def nonlin(x,deriv = False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))


X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([[0],[1],[1],[0]])

np.random.seed(1)

w0 = 2*np.random.random((3,4))-1
w1 = 2*np.random.random((4,1))-1

for j in range(100000):
	l0 = X
	l1 = nonlin(np.dot(l0,w0))
	l2 = nonlin(np.dot(l1,w1))

	l2_error = y-l2
	if(j%10000)==0:
		print("Err:"+str(np.mean(np.abs)))