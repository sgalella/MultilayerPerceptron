function accuracy = evaluate(NN, testData)
%
% Function:
% - evaluate: Check performance of the neural network with test data
%
% Inputs:
% - NN: Trained neural network (NeuralNet)
% - testData: Data used to test the network (cell of size 1x2)
%
% Outpus:
% - Accuracy: Percentage of correct answers of the neural network (double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the homonymous function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

% Initialize the vector with the answers and predictions
testResults = NaN(1,length(testData(2)));
prediction = NaN(1,length(testData(2)));

% Compute the feedforward run for the different test examples and assess if
% the output and the label match
for i = 1:length(testData{2})
    y_pred = feedforward(NN, testData{1}(i,:)');
    prediction(i) = ymap(y_pred);
    if isequal(prediction(i), testData{2}(i))
       testResults(i) = 1; 
    else
        testResults(i) = 0;
    end
end

% Average the results and compute the percentage
accuracy = 100*mean(testResults);



end

