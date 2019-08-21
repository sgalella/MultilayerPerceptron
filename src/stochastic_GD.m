function [NN, accuracy_epochs, total_cost] = stochasticGD(runSeed, NN, training_data, test_data, eta, mini_batch_size, epochs)
%
% Function:
% - stochasticGD: Computes the stochastic gradient descent for the network
%
% Inputs:
% - runSeed: Seed to generate stream to generate different training sets
% across minibatches (double)
% - NN: Initialized neural network (NeuralNet)
% - training_data: Data used to train the newtork (cell of size 1x2)
% - test_data: Data used to test the network (cell of size 1x2)
% - eta: Learning rate (double)
% - mini_batch_size: Training examples per mini batch (double)
% - epochs: Number of total epochs (double)
%
% Outputs:
% - NN: Neural network with optimized weights and bias (NeuralNet)
% - accuracy_epochs: Accuracy of the network along epochs (double)
% - total_cost: Cost for all the minibatches along epochs (double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the 'SGD' function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

% Create random stream using runSeed
s = RandStream.create('mrg32k3a','NumStreams', 1,'Seed',runSeed);

% Define the size of the training set (number of examples)
n_train = size(training_data{1,1},1);

% Define the number of total mini batches
num_mini_batches = n_train/mini_batch_size;

% Initialize the vector for the accuracy, cost per minibatch and total cost
accuracy_epochs = NaN(1,epochs);
cost_minibatch = NaN(1,num_mini_batches);
total_cost = NaN(1,epochs);

% Run the stochastic gradient descent
for epoch = 1:epochs
    
    % Create different minibatches with different training examples
    permutation = randperm(s,n_train);
    random_training_set_X = training_data{1,1}(permutation,:);
    random_training_set_Y = training_data{1,2}(permutation);
    mini_batches_X = cell(num_mini_batches,1);
    mini_batches_Y = cell(num_mini_batches,1);
    for j = 1:num_mini_batches
       mini_batches_X{j} = random_training_set_X((j-1)*mini_batch_size+1:j*mini_batch_size,:);
       mini_batches_Y{j} = random_training_set_Y((j-1)*mini_batch_size+1:j*mini_batch_size);
    end
    
    % Update the weights and bias of NN through the different mini batches
    for minibatch = 1:length(mini_batches_X) 
        [NN, cost_minibatch(minibatch)] = update_mini_batch(NN, mini_batches_X{minibatch},mini_batches_Y{minibatch}, eta);
    end
    
    % Compute the accuracy and cost of the epoch
    accuracy_epochs(epoch) = evaluate(NN, test_data);
    total_cost(epoch) = mean(cost_minibatch);
    fprintf('Epoch %d/%d. Accuracy: %.3f%%. Cost: %.3f.\n',epoch,epochs,accuracy_epochs(epoch),total_cost(epoch));
end

end

