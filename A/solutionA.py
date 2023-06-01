import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

training_data = np.genfromtxt('A/TrainingDataBinary.csv', delimiter=',')
testing_data = np.genfromtxt('A/TestingDataBinary.csv', delimiter=',')
x_training = training_data[:,:128] #use the first 127 columns of data for training
y_training = training_data[:,128] #use the last training column as the label
x_testing = testing_data[:,:128] #load in the testing data
x_train,x_test,y_train,y_test = train_test_split(x_training,y_training,test_size=0.2,random_state=100) #split training data into training and testing following the 80/20 rule

#grid search cv omitted for speed
rf_model = RandomForestClassifier(n_estimators=2000) #initialise model with tweaked hyperparameters to increase size of desicion tree ensemble
rf_model.fit(x_train, y_train) #fit random forest classifier to the training data
y_pred = rf_model.predict(x_test) #make predictions on testing data

y_pred_p = rf_model.predict_proba(x_test)[:, 1] #get the prediction's probabilities for each entry in the model
threshold = 0.4
for i in range(len(y_pred_p)): #reclassify values given the threshold
    if y_pred_p[i] >= threshold:
        y_pred[i] = 1
    else:
        y_pred[i] = 0 

#calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#generate and print the confusion matrix
conf_matr = confusion_matrix(y_test,y_pred)
print("Confusion matrix")
print(conf_matr)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matr, display_labels=rf_model.classes_,)
disp.plot()
plt.show()

#calculate and print the F1 score
f1 = f1_score(y_test, y_pred)
print("F1 score:", f1)

#visualise feature importance on a bar chart
plt.bar([x for x in range(128)],rf_model.feature_importances_)
plt.xlabel('Feature number')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()

y_pred_test = rf_model.predict(x_testing)

#output results
with open("A/TestingResultsBinary.csv","w") as f:
    i = 0
    temp = ""
    for res in y_pred_test:
        updated_list = (x_testing[i]).tolist()
        updated_list.append(int(res))
        fin_str = "\n"
        fin_str = str(updated_list) + fin_str
        fin_str = fin_str.replace("]","")
        fin_str = fin_str.replace("[","")
        f.write(fin_str)
        i = i + 1
    


