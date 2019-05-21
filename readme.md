# Survival of Passengers on Titanic using Machine Learning and Graphical User Interface

This project consists of a clean and polished Graphical User Interface (GUI) that interacts with 8 Machine Learning models and data visualization tools through the use of different Python libraries. At Rutgers, we learned about Python being a great general purpose language, which allows for great versatility for developers of all specialties. Therefore, we decided to take advantage of Python’s strong support for GUI development as well as its Data Science and Machine Learning capabilities. Using the complex RMS Titanic data set, which includes information about each passengers fate (survived/deceased) according to their economic status, fair, cabin, social class, relatives, gender, port of embarkment and age, we created 8 different Machine Learning models (Logistic Regression, Stochastic Gradient Descent, K Nearest Neighbor, Random Forest, Naive Bayes, Perceptron, Linear Support Vector Machine, Decision Tree) that learn from the data set and then perform accurate predictions of survival on the testing data provided by the user. In addition, we created an extensive GUI that allows the user to learn and interact with the training and testing data by displaying many different data plots and graphs as well as descriptions about the specifics (advantages and disadvantages) of each Machine Learning model. The user can interact with the GUI through selecting which model to run the testing data on, which then takes them to a screen displaying the prediction results of the testing data as well as the general model accuracy. The screen also includes various buttons that, when selected, display complex and attractive data visualizations on the testing data. The goal for this project was to get a good understanding of Python’s Data Science and Machine Learning support and to learn about GUI development and integration in Python. Upon completing the project, we had an increased appreciation for the power of Machine Learning and its potential as well as the customizability and complexities of GUI development. By partaking in GUI development, Data Manipulation/Visualization creation and Machine Learning development, this project is a clear representation of the power of the Python programming language and its overall integrability.

## Machine Learning Models Implemented:
- Logistic Regression
- Stochastic Gradient Descent
- K Nearest Neighbor
- Random Forest
- Naive Bayes
- Perceptron
- Linear Support Vector Machine
- Decision Tree


## Dependencies:
* Numpy
* Pandas
* SciKit-Learn
* SciPy
* Seaborn
* Matplotlib
* Tkinter
* Pillow


## GUI Preview:

![image](https://user-images.githubusercontent.com/39894720/58125916-b66c8700-7bdf-11e9-8e6b-8e9147e4f21d.png)

![image](https://user-images.githubusercontent.com/39894720/58125959-ca17ed80-7bdf-11e9-8b4b-c1def578ae03.png)

![image](https://user-images.githubusercontent.com/39894720/58125962-ce440b00-7bdf-11e9-9576-6e0bdb5d2659.png)

![image](https://user-images.githubusercontent.com/39894720/58125968-d308bf00-7bdf-11e9-8b76-6c9450c2eaaa.png)



## Installation:
1. Download and extract zip file

2. Create virtual environment inside FinalProject directory using
"python3 -m venv venv"

3. Run virtual environment in project directory using
"source venv/bin/activate"

4. Install required dependencies using
“pip install -r requirements.txt”

5. Run GUI using
"python GUI.py"

6. Exit virtual environment
"deactivate"

Note: User can input their own testing data to predict by changing the "test.csv" file with custom information. Do not delete first row of CSV as it contains column headers!


## Contributors:
Zain Siddiqui, Krupal Patel, Haneef Pervez

