# This program consists of clean and polished Graphical User Interface (GUI) that interacts with 8 Machine Learning models and data visualization tools through the use of different Python libraries. 
# The user can interact with the GUI through selecting which model to run on the testing data on, which then takes them to a screen displaying the prediction results of the testing data as well as the general model accuracy. 
# The screen also includes various buttons that, when selected, display complex and attractive data visualizations on the testing data.
import tkinter as tk
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk
import dataPreprocessing
import machineLearningModels
import pandas as pd

class root(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("1500x600+0+0")
        self.title("Titanic Disaster Platform")
        container = tk.Frame(self)
       # container.geometry("500x500")
        
        container.pack(side = "top", fill = "both", expand = True)
        container.grid_rowconfigure(0, weight = 3)
        container.grid_columnconfigure(0, weight = 3)
        
       # container.place(x = 500, y = 1500)
        self.frames = {}

        for F in (StartPage, Page1, Page2, Page3, Page4, Page5, Page6, Page7, Page8):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row = 0, column = 0, sticky = "snew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        load1 = Image.open('MLModelPics/bg2.png')
        render1 = ImageTk.PhotoImage(load1)
        img1 = tk.Label(self, image = render1)
        img1.image = render1
        img1.place(x = 0, y = 0)
        
        load2 = Image.open('MLModelPics/img1.jpg')
        render2 = ImageTk.PhotoImage(load2)
        img2 = tk.Label(self, image = render2)
        img2.image = render2
        img2.place(x = 65, y = 25)

        load3 = Image.open('MLModelPics/img2.png')
        render3 = ImageTk.PhotoImage(load3)
        img3 = tk.Label(self, image = render3)
        img3.image = render3
        img3.place(x = 1175, y = 25)

        labelA = tk.Label(self, text= "These three buttons will help you understand\nand visualize the type of people that were on the Titanic\n by showing graphs of some interesting correlations\n that occur with survival/death rates\nand other passenger attributes.")
        labelA.place(x = 1150, y = 300)

        labelB = tk.Label(self, text = "Each of our machine learning models employs an\n algorithm that will go through the large set\n of Titanic data and give a percentage\n of correctness of survival/death rates along\nwith a prediction array. Keep in mind,\n that each time the model will load up a completely\n new prediction percentage because\n the data is very large.")
        labelB.place(x = 50, y = 300)

        buttonA = tk.Button(self, text = "Train Distribution", command = lambda: machineLearningModels.trainclassDistr(machineLearningModels.train_df))
        buttonA.place(x = 1250, y = 400)

        buttonB = tk.Button(self, text = "Mean Fare Survival", command = lambda: machineLearningModels.trainMeanFareSurvival(machineLearningModels.train_df))
        buttonB.place(x = 1245, y = 450)

        buttonC = tk.Button(self, text = "Class Survival", command = lambda: machineLearningModels.trainClassSurvival(machineLearningModels.train_df))
        buttonC.place(x = 1260, y = 500)

        label = tk.Label(self , text= "Welcome to the Titanic Disaster Analysis Platform!\n\n Here you have the abilty to run algorithms and see data on the disaster that occurred in 1912 based on real passenger data.\n Our team has used machine learning and data processing techniques to provide several examples and predictions of surivival/death rates.\n\n This platform was created by Zain Siddiqui, Haneef Pervez, and Krupal Patel")
        label.place(x = 400, y = 150)
       
       # button2 = tk.Button(self, text = "useLog Regression", command = hello)
       # button2.place(x = 600, y = 150)
        button = tk.Button(self, text = "Log Regression", command = lambda: controller.show_frame(Page1))
        button.place(x = 500, y = 300)
        
        button2 = tk.Button(self, text = "Stochastic Gradient\nDescent", command = lambda: controller.show_frame(Page2))
        button2.place(x = 700, y = 300)
        
        button3 = tk.Button(self, text = "K Nearest Neighbor", command = lambda: controller.show_frame(Page3))
        button3.place(x = 900, y = 300)
        

        button4 = tk.Button(self, text = "Random Forest", command = lambda: controller.show_frame(Page4))
        button4.place(x = 500, y = 400)

        button5 = tk.Button(self, text = "Decision Tree", command = lambda: controller.show_frame(Page5))
        button5.place(x = 715, y = 400)

        button6 = tk.Button(self, text = "Perceptron", command = lambda: controller.show_frame(Page6))
        button6.place(x = 925, y = 400)

        button7 = tk.Button(self, text = "Linear Support\nVector", command = lambda: controller.show_frame(Page7))
        button7.place(x = 600, y = 500)

        button8 = tk.Button(self, text = "Gaussian Naive\nBayes", command = lambda: controller.show_frame(Page8))
        button8.place(x = 825, y = 500)

class Page1(tk.Frame):        
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)
                
                load1 = Image.open('MLModelPics/bg2.png')
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)

		
                load2 = Image.open('MLModelPics/LogRegression.jpg')
                load2 = load2.resize((450, 450))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2
                img2.place(x = 950, y = 75)

                labelSum = tk.Label(self, text = "Logistic regression is a classification algorithm.\nIt predicts the probability of an input belonging to a certain set by \nseparating data into two regions. Logistic regression is used when the response \nvariable will be binary, for example, pass/fail.")
                labelSum.place(x =100, y= 200)

             #   label = tk.Label(self, text= machineLearningModels.logRegression(machineLearningModels.train1, machineLearningModels.train2, machineLearningModels.test))
            #label.place(x = 700,y = 100)

                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, machineLearningModels.logRegression(machineLearningModels.train1, machineLearningModels.train2, machineLearningModels.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 700, y = 20)

                labelScroll = tk.Label(self, text = "Hover over the predicition box and scroll\n down to see the model accuracy")
                labelScroll.place(x = 660, y= 410)

                button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
                button1.place(x = 725, y = 500)

                buttonZ = tk.Button(self, text="Prediction Graph", command=lambda: machineLearningModels.groupPlot(machineLearningModels.predLog))
                buttonZ.place(x = 275, y = 400)


class Page2(tk.Frame):
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)
                
                load1 = Image.open('MLModelPics/bg2.png')
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)

                load2 = Image.open('MLModelPics/gradient-descent.png')
                load2 = load2.resize((450, 450))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2
                img2.place(x = 950, y = 75)

                labelSum = tk.Label(self, text = "Gradient Descent is an algorithm that minimizes a cost function. \nGradient descent is important in machine learning because it optimizes how well a machine \nlearning algorithm is working by minimizing that algorithm’s cost function. It works by calculating gradients\n and using those gradients to change weights for predictions.")
                labelSum.place(x =65, y= 200)

                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, machineLearningModels.SGD(machineLearningModels.train1, machineLearningModels.train2, machineLearningModels.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 700, y = 20)

                labelScroll = tk.Label(self, text = "Hover over the predicition box and scroll\n down to see the model accuracy")
                labelScroll.place(x = 660, y= 410)

                button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
                button1.place(x = 725, y = 500)

                buttonZ = tk.Button(self, text="Prediction Graph", command=lambda: machineLearningModels.groupPlot(machineLearningModels.predSGD))
                buttonZ.place(x = 275, y = 400)

class Page3(tk.Frame):
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)
                
                load1 = Image.open('MLModelPics/bg2.png')
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)

                load2 = Image.open('MLModelPics/kNearestNeighbor.png')
                load2 = load2.resize((450, 450))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2

                labelSum = tk.Label(self, text = "K-Nearest Neighbors is an algorithm that can be used for both classification and regression. \nIt works by taking in training data and then seeing which data points are close to a data point\n and then classifying that data point as part of the same class as the majority of the k-nearest \ndata points. An advantage of k-nearest neighbors is that it is usually pretty accurate and \nworks well for non-linear data. A disadvantage is that it has to store all the training\n data which can lead to memory and runtime issues.")
                labelSum.place(x =100, y= 200)

                img2.place(x = 950, y = 75)

                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, machineLearningModels.KNN(machineLearningModels.train1, machineLearningModels.train2, machineLearningModels.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 700, y = 20)

                labelScroll = tk.Label(self, text = "Hover over the predicition box and scroll\n down to see the model accuracy")
                labelScroll.place(x = 660, y= 410)

                button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
                button1.place(x = 725, y = 500)

                buttonZ = tk.Button(self, text="Prediction Graph", command=lambda: machineLearningModels.groupPlot(machineLearningModels.predK))
                buttonZ.place(x = 275, y = 400)


class Page4(tk.Frame):
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)
                
                load1 = Image.open('MLModelPics/bg2.png')
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)


                load2 = Image.open('MLModelPics/rForest.png')
                load2 = load2.resize((450, 450))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2
                img2.place(x = 950, y = 75)

                labelSum = tk.Label(self, text = "Decision trees that grow deep might lead to some incorrect results, random forests are a way to \ndeal with this. Random forests work by building a number of decision trees and merging them together\n and taking the averages of the test variables. One advantage of random forests is that it reduces\n the variance of using decision trees. A disadvantage is that making multiple decision \ntrees and merging them together might be run slow in some cases.")
                labelSum.place(x =75, y= 200)


                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, machineLearningModels.rForest(machineLearningModels.train1, machineLearningModels.train2, machineLearningModels.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 700, y = 20)

                labelScroll = tk.Label(self, text = "Hover over the predicition box and scroll\n down to see the model accuracy")
                labelScroll.place(x = 660, y= 410)

                button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
                button1.place(x = 725, y = 500)

                buttonZ = tk.Button(self, text="Prediction Graph", command=lambda: machineLearningModels.groupPlot(machineLearningModels.predForest))
                buttonZ.place(x = 275, y = 400)

class Page5(tk.Frame):
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)
                
                load1 = Image.open('MLModelPics/bg2.png')
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)

                load2 = Image.open('MLModelPics/DecisionTree.jpg')
                load2 = load2.resize((450, 450))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2
                img2.place(x = 950, y = 75)

                labelSum = tk.Label(self, text = "A decision tree is a structure that is used to predict the value of a target value based on certain input variables.\n Each node of the tree represents a decision that will affect the outcome of the target value. \nOne advantage of using a decision tree to predict behavior is that decision trees are easy to understand and follow.\n It is easy to follow the conditional logic that decision trees use.")
                labelSum.place(x =50, y= 200)


                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, machineLearningModels.dTree(machineLearningModels.train1, machineLearningModels.train2, machineLearningModels.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 700, y = 20)

                labelScroll = tk.Label(self, text = "Hover over the predicition box and scroll\n down to see the model accuracy")
                labelScroll.place(x = 660, y= 410)

                button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
                button1.place(x = 725, y = 500)

                buttonZ = tk.Button(self, text="Prediction Graph", command=lambda: machineLearningModels.groupPlot(machineLearningModels.predTree))
                buttonZ.place(x = 275, y = 400)

class Page6(tk.Frame):
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)

                load1 = Image.open('MLModelPics/bg2.png')
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)


                load2 = Image.open('MLModelPics/percepetron(neuron).jpg')
                load2 = load2.resize((450, 450))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2
                img2.place(x = 950, y = 75)


                labelSum = tk.Label(self, text = "A perceptron is used for classifying data. It is a linear classifier so it classifies \ndata into two categories. It works by calculating a weighted sum of input values and returning 1 \nif the sum is greater than a certain value or it returns 0 otherwise.")
                labelSum.place(x =100, y= 200)

                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, machineLearningModels.precep(machineLearningModels.train1, machineLearningModels.train2, machineLearningModels.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 700, y = 20)

                labelScroll = tk.Label(self, text = "Hover over the predicition box and scroll\n down to see the model accuracy")
                labelScroll.place(x = 660, y= 410)

                button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
                button1.place(x = 725, y = 500)

                buttonZ = tk.Button(self, text="Prediction Graph", command=lambda: machineLearningModels.groupPlot(machineLearningModels.predPer))
                buttonZ.place(x = 275, y = 400)


class Page7(tk.Frame):
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)

                load1 = Image.open('MLModelPics/bg2.png')
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)

                load2 = Image.open('MLModelPics/LinearSUPPORTvector.png')
                load2 = load2.resize((450, 450))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2
                img2.place(x = 950, y = 75)


                labelSum = tk.Label(self, text = "Support Vector Machine is a linear model that can be used for classification\n or regression. It works by creating a line or plane that separates \ndata into classes. It finds a line between points from the two classes and then maximizes the distance \nbetween the line and the points. SVMs usually have fast performances.")
                labelSum.place(x =75, y= 200)

                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, machineLearningModels.lSVM(machineLearningModels.train1, machineLearningModels.train2, machineLearningModels.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 700, y = 20)

                labelScroll = tk.Label(self, text = "Hover over the predicition box and scroll\n down to see the model accuracy")
                labelScroll.place(x = 660, y= 410)

                button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
                button1.place(x = 725, y = 500)

                buttonZ = tk.Button(self, text="Prediction Graph", command=lambda: machineLearningModels.groupPlot(machineLearningModels.predLSVM))
                buttonZ.place(x = 275, y = 400)

class Page8(tk.Frame):
        def __init__(self, parent, controller):
                tk.Frame.__init__(self, parent)

                load1 = Image.open('MLModelPics/bg2.png')
                render1 = ImageTk.PhotoImage(load1)
                img1 = tk.Label(self, image = render1)
                img1.image = render1
                img1.place(x = 0, y = 0)

                load2 = Image.open('MLModelPics/gNaiveBayes.png')
                load2 = load2.resize((450, 450))
                render2 = ImageTk.PhotoImage(load2)
                img2 = tk.Label(self, image = render2)
                img2.image = render2
                img2.place(x = 950, y = 75)


                labelSum = tk.Label(self, text = "Naive Bayes is a collection of algorithms that are based on the Bayes Theorem. \nIt works by classifying features of input data. It can then predict \nthe class of input by seeing which features match between \nclasses. Some advantages are that it is fast and easy to train. A disadvantage is that it assumes every\n feature is independent of the other which is not always true for every class.")
                labelSum.place(x =80, y= 200)

                textArea = tk.Text(self, height = 20, width = 15, wrap = tk.WORD)
                textArea.insert(tk.END, machineLearningModels.gNaiveBayes(machineLearningModels.train1, machineLearningModels.train2, machineLearningModels.test))
                textArea.configure(font=("Arial",12))

                scroller = tk.Scrollbar(self, orient= tk.VERTICAL)
                scroller.config(command = textArea.yview)
                textArea.configure(yscrollcommand= scroller.set)
                scroller.pack(side = tk.RIGHT, fill = tk.Y)
                textArea.place(x = 700, y = 20)

                labelScroll = tk.Label(self, text = "Hover over the predicition box and scroll\n down to see the model accuracy")
                labelScroll.place(x = 660, y= 410)

                button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
                button1.place(x = 725, y = 500)

                buttonZ = tk.Button(self, text="Prediction Graph", command=lambda: machineLearningModels.groupPlot(machineLearningModels.predBayes))
                buttonZ.place(x = 275, y = 400)

                
display = root()
display.mainloop()
