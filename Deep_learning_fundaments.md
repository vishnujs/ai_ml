Deep learning fundaments: Cognitive class
https://courses.cognitiveclass.ai/courses/course-v1:DeepLearning.TV+ML0115EN+v2.0/courseware/407a9f86565c44189740699636b4fb85/12eab34ec218468995e4d06566ef4a32/

Good link by IBM
https://www.ibm.com/support/knowledgecenter/ro/SSLVMB_24.0.0/spss/tutorials/fac_cars_tve.html

great researchers
	andrew NG, Goef Hinton, Yann Lecun, Yoshua bengio, Andrej karpathy

neural netweork takes input based on which it will perform complex calculations and then use the ouput to solve the problem

classification problems can be handled by Naive Bayes,SVM, Logistic regression, neural network
	a confidence score is the output; high confidence means : sick
		low confidence: health is normal
neural network are highly strucutural and comes in layers

input layer->hidden layers->output layer
it can be seen as spinning classifier in a layered web
each nodes act as its own classifier
the score of one layer is passed on to the next node for activation
this series of events starting form input to output is called forward propogation
first classifier: the perceptron - need further reading
it was found out the accuracy of output can be increaced with layers and hence MLP
MLP->multi layer perceptron

each input have the same classifier so giving same input should produce same output right? The answer to it is no
	for each input is modified with unique weights and bias

prediction mainly depends on the weihgt and bias
we need to make the output close as the actual output to increase the accuracy

the cost is the differnce of actual ouput and predicted output.

the reason of using the deep learning for classification is due to pattern complexity

nural nets outperform all other complex patterns classification 
	this is made happen because complex patterns are broken down into simpler ones in nural nets
	simpler solutions are then combined to a new one
first decide what to find
	pattern
	classification

for nlp:sentiment analysis : Recurrent net or recursive nural tensor network
	any model at charecter level used RNTN
for image recognition
	Deep belief network or convolutional network

for object recognition
	use convolutional netw or RNTN

for speech recognition
	use recurrent netw
DBN & RELU 
	are good choices for classification
for timeseries
 	recurrent netw

while training the data using back propogation two errors occurs
	vanishing gradient or exploding gradient

gradient is the measure of rate at which the cost changes for any change in the weight or bias

backpropogation is used for training a neural netw1
gradient at a particular level is the product of all gradients  at previous layer

things changed on the paper published by hinton ,lecun and bengio
restricted bolzman machin


Deep learning using tensorflow
	
	THe main deep learning methods discussed here are:
		Convolutional Neural network
		Recurrent Neural Network
		Restrictive Boltzman Machines
		Deep belief network
		AutoEncoders

	in traditional models we will be preforming the feature extraction and this will be given to a machine learning algorithms which in turn will be a model that will accurately predict a new input

	in the case of CNN-convolutional neural network will find itself the features(based on complex mathematical solutions) and will classify based on that and hence no external feature selection is requiered.

	RNN:Recurrent Neural Network are those which can predict next values based on previous input. It requires large text corpus ie huge data set for better accurate prediction. Its main application inclues 

	The RBM-restrive Boltzman machine as a single unit is not product but once deployed in layers they are very effective for propogation problems.RBMs are best suited for unsuperviced tasks such as feature extraction,pattern recognition and handling missing values.

	Deep Belief networks are very effective in solving the backpropogation that can cause local minima and vanishing gradient.They have multilayers of RBS. it works like MLP(multi layer perceptron:which needs a non linear activation function and used backward propogation).
	DBNs are used for classification of mainly images, very accurate descriminative classifier, small dataset is required

	AutoEncoders are more like DBNs with a different architecture. They convert a single image into set of short codes and are then used for reconstructing the image thus extracting features. They are used for feature extraction. AutoEncoders take the image as input and then convert into short codes (which can be used for dimentionality reduction), These then will be used for detecting featurs by reconstruting them.For eg for face detection initially it will try to detect the edges and then it will detect the less complex parts such as the nose.


CNN
	it is used for solving two main challenges
		detect and classify objects into catogories
		independent from pose,illumination,scale,conformation(shape),clutter(images in a group)
	it works just like how our brain detects objects
		the input is initially devidided into simple feature
			eg a building will consits of horizontal and vertical lines
		object part detection
			comining theese features it will try to detect the building
			more complex inbuilt structures will be detected like doors and windows
	based on these persisted features it will detect a building

	basic procedures can be listed as
		input
		primitive features
		Object parts
		object

	in normal shallow network we need to define the feature which is inefficient and time consuming and also such models are context specfic
	CNN automatically detects the features
	MINST dataset contains 60000 hand written images
		each letters are fixed sized images and also cetralized
			?? how to make such dataset is the most important task of all as it determines the effectiveness of the data
			?? need to find out any automation for the same, tools or other methods

	main steps for handwriting recognition
		preprocessing
		training
		inference and deployment

	training layers
		convolutional layers
		pooling layers
		fully connected layers

		"The convolutional layer applies a convolution operation to the input, and passes the result to the next layer. The pooling layer combines the outputs of a cluster of neurons in the previous layer into a single neuron in the next layer. And then, the fully connected layers connect every neuron in the previous layer to every neuron in the next layer. In the next video, you will learn more about"

RNN
	Recurrent neural netweorks
	problem with sequential data
		sequential data is something that is dependent upon the previous data
			eg:stock market the value of current is in relevance with the value of previous data

		in traditional neural network which uses feed forward will take input process it in hidden layers with weights and biases
		they takes input as non sequential and hence take into account only one input at a time and then input after input

		some of the scenarios such as weather, stock market, sentence
			in the case of weather the current weather has an influence in the comming weather
		RNN can solve this problem




Dimentionality reduction
	main questions
		intrested to see how observations hang togather
			market segmentation
			types of observation
			grouping observations togather

		intrested to see how variables hang togather
			varibales may describe similar things
			what is the underlying similarity
			grouping varibles

			it is hard to find the depedent varibles

		exploratory factor analysis and principal component analysis
			exportory factor analysis
				multivariate statistics
					simultanious observation and analysis of more than one outcome variables
						outcome variables are those variables that are dependent variables measured by changing independent varibales.
				used for unconvering underlying strucure of relatively large sets of varibales
				used factor analysis for finding relationship between measure variables
					factor analysis varibility among obderved,correlated varibales in terms of factors(lower number of unobserved factors)
					they help to find latend variables(variables that are not directly observed)


		high correlations among variables

	dimention reduction solutions
		to reduce the dimension of the population
			perform cluster analysis
		to reduce the dimension of the construct
			principal component analysis
			exploratory factor analysis
	dimension of construct
		we need to partition the data in columns(hence noSQL most suitable)
		and underlying the similarity between different observations

	to reduce the number of variables

	higher order dimensions are permutations of lower order dimensions

	some varibles are highly corelated with each other and some are not

	PCA will look into the correlation matrix and if there is high correlation with the variables it will suggest to group them as a single component

	EFA takes an intresting assumption that there are undelying leads and hypothetical factor that you have not measured and can be used to describe all of the varaibles
		it is based on relations the factor that we have taken into account has direct or indeirect relation
		so pictorially they are represented as with straight lines with arrow showing the directions
		if there a variable without an impact it will be shown with dashed lines
	

	differenc between PCA and EFA
		PCA explains total variance-EFA explains common variance 
		PCA identifies measures that are sufficiently similar to each other to justify combination-EFA captures latent constructs assumed to causethe variance		
			latent constucts are those measures that cannot be directly measured. They are not directly obsevable. By using indicators that represent the underlying construct
		most behavioral questions that involve factor analysis are theoritically akin to EFA

	loading
	communality
	uniqueness

	





	
