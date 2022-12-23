# Typescript Logistic Regression (from scratch)

This is an implementation of logistic regression (machine learning) in Typescript. The file includes a number of indirectly-related helper functions which can find the mean of and normalize a 2D matrix, among other things. I'm 80% certain it works (some parts of it don't have tests yet). 

The main ```logistic()``` function takes a 2D matrix of inputs and a 1D vector of outputs (along with some other optional parameters, see the code for details). It returns a function that predicts the label (0 or 1) of a new observation (1D vector of same length as the # of columns the 2D matrix of training inputs has).

I made this primarily to reinforce my understanding of basic machine learning concepts, but if you want to use it to run logistic regression on the web, that's OK with me.

