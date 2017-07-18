import random #for initializing random weights and random biases
import numpy as np #for doing matrix calculations
import matplotlib.pyplot as plt
import matplotlib.colors as pltclr
import math #for isnan
import copy


#some linear function
if 1:
	T_DATA_IN = []
	T_DATA_OUT = []
	points = 50
	for a in range(points):
		noise = random.randint(-100,100)/5000.0
		T_DATA_IN.append([-float(a-points/2)/points])
		T_DATA_OUT.append([float(a)/points+noise])
	CASE = 'regression'
	HIDDEN_LAYERS = [8] #count of nodes for hidden layers. e.g. [2,3] or [9,2,2,2]
	MAX_ITERATIONS = 100  #to prevent non-converging ininite loops
	ERROR_THRESHOLD = 0.6 #when the error is small enough. The larger the noise the larger the threshold
	LEARNING_RATE = 0.3
	SIGNMOID_EXP_MUL = 1 #e.g. 0.5 or 5; higher values here result in curvier models and bigger training steps.
	MOMENTUM = 0.3 #to overcome local minimas
	REINITIALIZE_THRESHOLD = 4 #error, at which the weights and biases should be reshuffled

#quadratic function
if 0:
	T_DATA_IN = []
	T_DATA_OUT = []
	points = 20
	for a in range(points+1):
		T_DATA_IN.append([float(a-points/2)/points])
		T_DATA_OUT.append([(float(a)/points-0.5)**2*4])
	CASE = 'regression'
	HIDDEN_LAYERS = [20,20]
	MAX_ITERATIONS = 600
	ERROR_THRESHOLD = 0.6
	LEARNING_RATE = 0.2
	SIGNMOID_EXP_MUL = 5
	MOMENTUM = 0.2
	REINITIALIZE_THRESHOLD = 10

#classification: from -0.3 to 0.3: 0,1,0
#		 from -inf to -0.3: 1,0,0
#		 from 0.3 to inf: 0,0,1
if 0:
	T_DATA_IN  = [[-0.9,1],[-0.7,1],[-0.5,1],[-0.3,1],[-0.1,1],[0.2,1],[0.4,1],[0.6,1],[0.8,1],[1,1]]
	T_DATA_OUT = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]
	CASE = 'classification'
	HIDDEN_LAYERS = [12,12]
	MAX_ITERATIONS = 1000
	ERROR_THRESHOLD = 1.2
	LEARNING_RATE = 0.1
	SIGNMOID_EXP_MUL = 3
	MOMENTUM = 0.3
	REINITIALIZE_THRESHOLD = 14


T_DATA_IN = np.array(T_DATA_IN).astype(float)
T_DATA_OUT = np.array(T_DATA_OUT).astype(float)
#this defines the shape of the network. Training data will match the shape
LAYER_SIZES = [len(T_DATA_IN[0])]+HIDDEN_LAYERS+[len(T_DATA_OUT[0])]
#some usefull things to make it simpler
LAYER_COUNT = len(LAYER_SIZES)

#this for-loop creates for each layer a weight matrix, initialised with random values from -1 to 1
#weights is all weights. weights[i] is the weights for one layer.
weights = []
autosaveweights = []
changes = []
biases = []
#iterate through all layers

def init():
	global weights
	global changes
	global biases
	for layer_nr in range(LAYER_COUNT)[1:LAYER_COUNT]:

		#initialize weightmatrix for a given layer
		weights.append([])
		changes.append([])

		#number of nodes from the previous layer is the rowcount of the weight matrix
		#number of nodes from the current layer is the colcount
		row_count = LAYER_SIZES[layer_nr-1]
		col_count = LAYER_SIZES[layer_nr]

		#changes will be used to access previous changes on a given node to apply momentum
		changes[layer_nr-1] = (np.zeros(shape=(row_count,col_count)))

		#fill weights matrix with random numbers from -1 to 1
		weights[layer_nr-1] = (np.random.randint(-100,100,(row_count,col_count))/100.0)

	#create bias array that has to match the networks shape
	biases = copy.deepcopy(LAYER_SIZES)
	for l in range(len(biases)):
		biases[l] = (np.random.randint(-100,100,(LAYER_SIZES[l]))/100.0)

def reshuffleweights(factor):
	global weights
	global biases
	#factor from 0 to 1
	#1 for complete reset, 0 for no change at all

	#reshuffle the weights a little bit
	for layer_nr in range(LAYER_COUNT)[1:LAYER_COUNT]:
		row_count = LAYER_SIZES[layer_nr-1]
		col_count = LAYER_SIZES[layer_nr]
		#fill weights matrix with random numbers from -1 to 1
		weights[layer_nr-1] += np.random.randint(-100,100,(row_count,col_count))/100.0*factor

	#reshuffle the bias a little bit
	#for l in range(len(biases)):
	#	biases[l] += np.random.randint(-100,100,(LAYER_SIZES[l]))/100.0*factor
	

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x*SIGNMOID_EXP_MUL))

def dsigmoid(x): #x is already sigmoided
	return x-(x**2)

#forward_propagate. v is a vector from the input layer
def fprop(v):

	#to be able to do .dot
	v = np.array(v)

	#put v in interims here, which is the initial input-node value
	interims.append(v)

	#start at index 1, because index 0 is the input v and therefore no need to calculate index 0
	for layer_nr in range(LAYER_COUNT)[1:LAYER_COUNT]:
		#v has always the shape as the layer "layer_index",
		#thanks to the matrixmultiplication from the previous iteration

		#calculate; multiply the layers with the weights one by one
		if len(v) != LAYER_SIZES[layer_nr-1]:
			print "ERROR, len(v) !=",LAYER_SIZES[layer_nr-1],";",v

		v = v.dot(weights[layer_nr-1])
		#v is the matrix of interims/output-results

		#now transform the fproped_value according to the activation function
		#iterate over all values in the input matrix.
		#in case of a output node, don't use sigmoid anymore. Use a linear output instead, but clip it at -1 and 1
		if layer_nr != LAYER_COUNT-1:
			for j in range(len(v)):
				v[j] = sigmoid(v[j]+biases[layer_nr][j])

		interims.append(v)

	#v will now has the form of the output layer,
	#because the last matrix-multiplication (.dot()) with the
	#last weight matrix will result in it
	return v

def test(x):
	x = max(-1,min(x,1))
	print x

plotcolor = [0.5,0,0] #first hsv color should be black
def plot():

	plotcolor[0] = (plotcolor[0]+1.0/MAX_ITERATIONS)%1.0
	lower_bound = min(T_DATA_IN.flat)
	upper_bound = max(T_DATA_IN.flat)
	step = 0.03 #set this to a lower value to get a smoother curve
	input_value = lower_bound
	x1 = []
	x2 = []
	while input_value < upper_bound:
		fproped_value = fprop([input_value])
		x2.append(fproped_value)
		x1.append(input_value)
		input_value += step

	#result
	plt.plot(x1,x2,
			marker = '',
			linestyle = '-',
			color = pltclr.hsv_to_rgb(plotcolor))

	#after the first line has been plottet in black, put some color in for the next lines
	plotcolor[1] = 0.5
	plotcolor[2] = 1
	#trainingdata

#train the neural network
#initialize the error in a way which does not prevent the loop from running
init()

total_error = ERROR_THRESHOLD+1
previous_total_error = -total_error
previous_gain = 0
interims = [] #needed for the backpropagation algorithm
iterations = 0

print "training..."
#stop when error is small enough or the computation takes too long
while iterations < MAX_ITERATIONS and total_error > ERROR_THRESHOLD:
	#iterate over the training data

	#if the error is very large and no autosave to roll back to available, start over and reinitialize random weights
	if total_error > REINITIALIZE_THRESHOLD:
		if autosaveweights == []:
			print "error:",total_error,">",REINITIALIZE_THRESHOLD,"| Reinitializing"
			init()
		else:
			print "Error became too large | reverting to earlier state and reshuffle"
			weights = copy.deepcopy(autosaveweights)
			reshuffleweights(0.05)

	#check if the network should start over again or if it's finished in case of convergence
	gain = abs(previous_total_error - total_error)
	reshuffle_threshold = total_error/6000.0
	if(gain < reshuffle_threshold and previous_gain < reshuffle_threshold):
		if(total_error > (ERROR_THRESHOLD*2)):
			reshuffle = round(total_error*100*5)/100
			print "Converging | Reshuffling by",reshuffle,"%"
			reshuffleweights(reshuffle/100)
		else:
			print "The network converged"
			break
	previous_gain = abs(previous_total_error - total_error)
	previous_total_error = total_error

	#at this point interims is broken because the plot() function called compute very often
	#but it's going to be reset before it can break anything. (just to prevent confusion)

	iterations += 1
	total_error = 0

	for i in range(len(T_DATA_IN)):

		#backpropagate
		interims = [] #reset interims

		target_value = T_DATA_OUT[i] #target for the NN to calculate
		fproped_value = fprop(T_DATA_IN[i])

		#create shape of network in delta
		delta = copy.deepcopy(LAYER_SIZES)
		for l in range(len(delta)):
			delta[l] = np.zeros(delta[l])


		#for each output node compute the delta
		for k in range(LAYER_SIZES[-1]):
			#some stuff for the total error:
			total_error += abs(fproped_value[k]-target_value[k])
		error = fproped_value-target_value
		delta[-1] = sigmoid(fproped_value)*error

		#for each hidden node calculate the delta
		for i in range(LAYER_COUNT)[0:-1][::-1]: #delta for last layer is already calculated; start at the end
			#i points to the current layer (index)
			for n in range(LAYER_SIZES[i]):
				#backpropagate
				delta[i][n] = dsigmoid(interims[i][n]) * delta[i+1].dot(weights[i][n])

		#update weights
		#"node_lpa" means "node-index in layer plus a"
		for layer in range(LAYER_COUNT-1):
			for node_lp0 in range(LAYER_SIZES[layer]):
				#update bias, no interaction with other layers -> layer
				biases[layer][node_lp0] -= LEARNING_RATE * delta[layer][node_lp0]
				#update weights, needs to check interims of next layer's nodes -> layer and layer+1
				for node_lp1 in range(LAYER_SIZES[layer+1]):
					#momentum
					previous_change = MOMENTUM * changes[layer][node_lp0][node_lp1] #change of weight from node_lp0 to node_lp1 
					change = LEARNING_RATE * delta[layer+1][node_lp1] * interims[layer][node_lp0] + previous_change
					changes[layer][node_lp0][node_lp1] = change
					#update
					weights[layer][node_lp0][node_lp1] -= change

	if CASE == 'regression':
		plot()

	if (100.0*iterations/MAX_ITERATIONS)%5 == 0:
		print "e:",round(total_error*1000)/1000,"\tbreak:",iterations*100/MAX_ITERATIONS,"%"
		if total_error < REINITIALIZE_THRESHOLD:
			autosaveweights = copy.deepcopy(weights)

if iterations == MAX_ITERATIONS:
	print "Training aborted because max_iterations was reached. Error:",total_error
else:
	print "Training finished. Error:",total_error

#training finished
#do some matplotlib plotting to show the neural-network's function

if CASE == 'regression':
	plot()

	#plot the training-datapoints
	#with transposing make sure, that no 2D input is put into this
	plt.plot(T_DATA_IN.T[0].T.flat,T_DATA_OUT.flat,
			marker = 'o',
			linestyle = '')
	plt.xlabel("input")
	plt.ylabel("forward-propagated")
	plt.title("training result graph")
	plt.margins(0.1)
	plt.show()

if CASE == 'classification':
	np.set_printoptions(formatter=dict(float=lambda t: "%5.2f" % t))
	for idatapoint in range(len(T_DATA_IN)):
		datapoint = T_DATA_IN[idatapoint].astype(np.float)
		out = fprop(datapoint).astype(np.float)
		print "in:",datapoint,"\tout:",out,"\ttarget:",T_DATA_OUT[idatapoint]

quit()



















