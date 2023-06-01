import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

training_data = np.genfromtxt('B/TrainingDataMulti.csv', delimiter=',')
testing_data = np.genfromtxt('B/TestingDataMulti.csv', delimiter=',')
x_training = training_data[:,:128]
y_training = training_data[:,128]
x_testing = testing_data[:,:128]

#in the training data there are:
#3000 0
#1500 1
#1500 2
#a for loop was used to count these values but it was omitted for code clarity

x_train,x_test,y_train,y_test = train_test_split(x_training,y_training,test_size=0.2,random_state=100) #split training data into training and testing following the 80/20 rule

class_weights = {0:1,1:4,2:4}
rf_model = RandomForestClassifier(class_weight=class_weights,criterion="entropy") #initialise model with custom class weightings and a specialised criteria for the algorithm
rf_model.fit(x_train, y_train) #fit random classifier to the training data
y_pred = rf_model.predict(x_test) #make predictions on testing data

#calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#generate and print the confusion matrix
conf_matr = confusion_matrix(y_test,y_pred)
print("Confusion matrix")
print(conf_matr)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matr, display_labels=rf_model.classes_)
disp.plot()
plt.show()

#calculate and print the F1 score
f1 = f1_score(y_test, y_pred,average="weighted")
print("F1 score:", f1)

#visualise feature importance on a bar chart
plt.bar([x for x in range(128)],rf_model.feature_importances_)
plt.xlabel('Feature number')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()

y_pred_test = rf_model.predict(x_testing) #predict the testing labels
    
#output results
with open("B/TestingResultsMulti.csv","w") as f:
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
