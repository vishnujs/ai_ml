main types of machine learning methods
	classification
	clustering
	regression
	dimentionality reduction


given the data where we need to come up with a particular relation between given data are called regression

linear regression
	it is one of the fundamental machine learning technique

	y=3x+2 where x is an independent variable and y is a dependednt variable

	z= 5x^2+y where x and y are independent varaible while z is an independent variable


	regression is modelling a relationship between dependent and independent varibles for prediction

	simple linear regression/ univariant linear regression: has only one independent varaible

	militple linear regression or multivariate linear regression: more than one independent variable

```

	import numpy asnp
	np.randon.seed(42)

	X= 2 * np.rand(100,1)

	y - 4 + 3 * X + np.random.randn(100,1)

```

	we need to construct a line that represents the most of the points

	linear regression can handle only straight lines
	for 3 dimention  it will be a plane

	hyper planes that are more than 3d

	those plots that are not in the line we need to find the distance between those points
		they are errors
	we will have to evaluate the cost for removing those errors

	what are the parameters that are required?
		y-intercept
		slope


		y=c+mx equation of line
		y hat = theta 0 + theta 1 x

		theta 0 is intercept
		theta 1 is the coefficient

		multiple linear regression
		y hat = theta 0 + theta 1 x + theta 2 x2+...+theta n xn

		y hat is the predicate

	vectorized general form

		thata transpose x
			y hat = h theta(x) = theta transpose.x

		theta = thata 0  theta 1 .....
		theta trnspose will be multipled with x values where x = [x0 x1 x2 x3 ...]



		y hat (i){line} - y(i) {green dot}

		SE(X,h theta) = zigma i=1 to m (y hat(i) - y (i)) the whole squared

		y hat (i) is the predicted value
		
		y (i) is the actual value



	MSE(mean square Error) :-  where we will devide with the number of sample data
	square root we will get root mean square error


	cost function is the expression 
	we need to have minimum value for that particular expression


	we have x and y .... we need to get the minimal value for theta

	for finding the minimum value of a function
		we need to take the first derivative
		then we need to take the second derivative


	matrix differenciation



	normal equation
		theta hat = (X trnspose .X)^-1 . Xt.y
		for getting the minimal theta value

	closed form solution

		y is the vector of target values


	zero inverse function

	rsquare evaluation


Gradient descend
	matrix inversion is a very costly operation
		O(n^2,3) -> O(n^3)

	an alternate way to determine the theta value which avoids matrix inversion

	most widely used for deep learning
		where the numbere of training data is huge

	inside backpropogation gradient descend is used

	as x incerases y  is decreasing which gives a negtive function

	f prime (10)

	convex function 
		line constructed on any two points in the curve will not intersect anywhere in the curve

	f = 1/2 * pow(x,2)
	f dash (x) = x

	if derivative is a negative value then it is below the curve
	if derivative is outside but under the line


	
	idea behind the gradient descent
		when the derivative value is negative increase the value of x
		when the derivative value is positive decrease the value of x

	when we are dealing with mulivariable generalisation
	 where there are multiple functions for finding the derivative
	 	for each varible we have to obtain the partail derivative

	 jacobian
		 	if the output is another vector
		 	it is the collection of first derivative of the functions

		 cos function is sometimes written as j where j is for jacobian

	when we collect second derivative then we het

		Hessian
			it is a huge matrix
			it is not normally used
			it tells the curvature of the surface
			ML practically it is not feasible


	start with random inistialisation of data
	calculate error using cost funcion
	make small change to the parameter
	agian calculate error
	repeat until the error converage is minimal

	if the derivative is negative then the vlaue will be increased

	
	challenges

		what will happen when have a ueven surface
			local minimum which one of the low point but not the best
			global minimum which is the actual lowest point
			platea where optimisation will be very slow for each small value there may be minimal differecce

	feature scaling
		its sacles all values between 0 and 1

	standardisation
		based on standard deviation and mean
		standardisation is most preggered

	Types of Gradient

		batch
		stochastic
		mini batch


		batch gradient descent
			it uses the whole training set for parameter optimisation
				



