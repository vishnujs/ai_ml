machine learning

	deep learning will handle feature extraction with classification

	https://developers.google.com/machine-learning/crash-course/ml-intro

	colab
		change the runtime gpus and tpus can be configured from runtype which can be selected form the remove restart

regression,classification adn clustering
	k-means
		hierarchical clusterinf for the case when the value of k is not required to be given

ML surprise
	the lecture talks about the definition of conventional way of programming compared with the new ways how it need to be.
	In ML the programming metholody will decide what to do from a set of examples.

The ML surprise 
	Defining KPI's
	Collecting Data
	Building infrastructure
	Oprimising algorithm
	Integration

	in Ground reality where there is more probability for success it is with involving more time in data prerp and other processes where a particular data can be run and rerun multiple times.
	So collecting data , building infrastrucure, Integration is the most time they spend on.


we provide data and answers to the ML model will provide the rules.
********************************************************************************************************************************************************************
19/06/2019

artificial intellegence->machine learning -> deep learning

	superviced machine-learning
		prediction models
			regression models
				logistic regression
					output is yes or no
					linear and multiple regression
					non linear regerssion
					other regressions:poisson regression 

					labels nad feature
						y is the label
						x is the predictor or feature
							eg y=mx+c
							based on multiple feature there will be multiple values
								if there is no y then the data will be unlabelled
					labelled data used to train the model
					model,training,inference

					linear regression is continous

					overfitting and underfitting


					Y-Predicted model
					B:Bias
					w:weight
					x is the 

					linear regression
						method of least squares
							estimate the best fitting
						multiple linear regression

					calculate the bias

					two phases
						training 
						loss is the penality for a bad prediction
							cost: mean square error

						by finetuning the weights we can obtain a minimial loss/cost
							gradient descent loss funciton to get the miniaml loss
							curve is showing the loss function also called the trainig phase
								learning rate:gradient * step
									vanishing gradient will occur if provided with large learning rate

								optimisers
									stochastic gradient
										traverse through the curve to get the minimal value
										compute the same with some random values not going through all values to minimise the computation
			epoch is one iteration

			regression analysis
			classification of model predict discrete values



03/07/2019
	intro to tensor flow
		data flow graph
		c++ with python interface
	Keras
	Tensorflow
		a graph protocol buffers
		a runtime for execution
	code structure
		graph mode
		eager mode
		tensors(data)
		edges are tensors:
			0d-scalar
			1d-vector
			2d-matrix
			edges rep numerical data
		constants are tensors not changable
		varibles:tensors inititalized as variables

		placeholders
		setting processor affinitiy
		imperative programming enviornment
	kafka and tf used togather
		comprehencive machine learning framework
	Tensorflow lite
		

09/07/2019

loass and optimisation
	stochastic gradient descent
	mini-batch gradient descent

	problems	
		overfitting
		underfitting
	Loss function
		Regression loss
			hyperparameters
			Mean Absolute Error/ L1 loss
			L1 less sensitive to outliers
			Mean Square Error/Quadratic Loss/L2 loss
			Mean Bias Error
		Classification loss
			Cross Entropy loss
			SVM
	Optimisers
	Gradient Descent
	weight initialization
			global min and local min
		polynomial usually have mutiple min value sin gradient descnet
	learning rate
		if it is high it might mistakely take non minimum values
	SGD
	mini batch gradient descent
	variants to gradient descend
		adaptive GD
		RMSprop Optimizer

	Reducing loss
	
17/07/2019

neurons
	perceptron
	multilayer perceptron

scaling neurons - wider and deeper
forier transform will convert each of them will be added	

24/07/2019

Generalisation
	
29/07/2019

validation data should never be used for training purpose as it could make the model efficient with great accuarcy but will perform poorly.

regularisation for addressing overfitting
adding a penality term to the cost function: aso called lambda value

types of regularisation
	l1
	l2
	drop out regularisation
	early stopping

	can be used for controlling the effect of loss function
	raising will reduce accuracy

	lasso regularisation(l1)
	ridge regularisation(l2)

	lasso used for millions of features
		reduces some of the features
	ridge

	drop out regularisation	
		ignores units


does it have any link with reinforcement learning where it is reward based
is this the dimentionality reduction
	
	early stopping
		tf.keras.callbacks.EarlyStopping
			monitors and if no improvement it will stop it
will there be any cases when the improvment happens after the threshold that we provide
	
	priciple component analysis, 			

07/08/2019

anything beyond a 3 zigma is outliyer


14/08/2019

Classification vs clustering

	classification
		superviced
		model construction
		model usage
		make a ruleL:lisp was used for such type of programmign

	clustering
		unsuperviced
cross entropy loss
	in any classification algorithm the entropy loss is condiered as the measure of performance
n g
	clustering
		centroid based clustering
		hierarchical based clusetering

		cluster distance measure
			single link
			complete link
			average

			disadv
				ourliers should be removed from the centroids
			for measuring how good a clusetering is:
				inter class
				intra class

09/10/2019
	explainable ai
		XAI

	we should be eliminating the human bias

	Fairness (Unbiasedness)				
