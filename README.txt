Using test.py Program To Predict Labels of Test Dataset

1. Download trained_model.pkl and test.py on your machine
2. trained_model.pkl is already trained model for given test scenario
3. Load test.py in your editor
4. Function, test_char(classifier, data) has two inputs
- (A) Trained model
- (B)  Test data in .pkl format
5. Enter path of trained_model.pkl and test data
6. Run the code
7. It will give vector with predicted labels
8. Labels values represents following characters
[1 = a, 2 = b, 3 = c, 4 = d, 5 = h, 6 = i, 7 = j, 8 = k and -1 = unknown]
9. We implemented extra credit part using probability function of svm. We are assigning label = -1 if our prediction for each class has less than 0.25 probability.
