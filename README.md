Before reading the following description, I'm here to appreciate the author of CDBN with MATLAB version firstly. Based on the author's achievement, there are some adjustments in the model of CDBN with binary type.

# Convolutional Deep Belief Networks with hyperparameter tunning using gridsearchCV

The layer part of the original program, 'DemoCDBN_Binary_2D.m', is unflexible. It is fixed in 2 layers. If you want to change the number of layers, you need to add a block of lines for each layer. Besides, hyperparameters are set directly here that makes the hyperparameter tuning work not easy to execute. This program uses a map object to save the hyperparameter for tuning using grid search and adds cross validation to robust the result of the model. To be more efficient, the program also uses MATLAB Parallel Computing Toolbox for parallelization.



## Run the program
* run 'setup_toolbox.m';
* run 'test_CVgrid_CDBN.m' 
 
            

If there're any problems or suggestions, welcome to contact me (judy79802002@gmail.com), thank you.

