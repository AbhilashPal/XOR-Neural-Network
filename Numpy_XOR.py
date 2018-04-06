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

	l2_error = Y-l2
	if(j%20000)==0:
		print("Err:"+str(np.mean(np.abs(l2_error))))

	l2_delta = l2_error*nonlin(l2,deriv=True)
	l1_error = l2_delta.dot(w1.T)
	l1_delta = l1_error*nonlin(l1,deriv=True)

	w1 += l1.T.dot(l2_delta)
	w0 += l0.T.dot(l1_delta)

print("After Training:")
print(l2)
