import pandas  as pd
import numpy as np
import tkinter as tk
import csv
from tkinter import messagebox
from tkinter import *
import cv2



def damage():
    window =tk.Tk()
    window.title("FINAL YEAR PROJECT")
    window.configure(background='seagreen1')
    window.geometry('1500x1000')


    x_cord = 75;
    y_cord = 20;
    
    message = tk.Label(window, text="DAMAGE DETECTION" ,bg="seagreen1"  ,fg="black"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
    message.place(x=170, y=20)
    message3 = tk.Label(window, text="" ,bg="white"  ,fg="black"  ,width=30  ,height=1, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
    message3.place(x=500-x_cord, y=500-y_cord)

               
                       

    def yolo():
        net = cv2.dnn.readNet("yolo-coco-data/yolov3_training_2000_L.weights", "yolo-coco-data/yolov3_testing_L.cfg")
        classes = []
        with open("yolo-coco-data/classes_L(D).names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))



        cap =cv2.VideoCapture(0)

        co=1
        if(co==1):
            _, img=cap.read()
            height, width, channels = img.shape


            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)


            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                  
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
     
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            #print(indexes)
            counter = 1
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                cv2.imshow("Image", img)
                key=cv2.waitKey(0)
                if key==27:
                    break


                if len(indexes) > 0:
                    for i in indexes.flatten():
                        #print('Object {0}: {1}'.format(counter, classes[int(class_ids[i])]))
                        res=format(classes[int(class_ids[i])])
                        message3.configure(text=res)

        co=co+1
        cap.release()     
        cv2.destroyAllWindows()

        
    def quit_window():
           window.destroy()




    trainImg = tk.Button(window, text="START TO DETECT AND PREDICT", command=yolo  ,fg="black"  ,bg="cyan3"  ,width=35  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=100)


    quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="cyan3"  ,width=10  ,height=2, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    quitWindow.place(x=1000, y=735-y_cord)

    window.mainloop()
    



def pred1():
    window =tk.Tk()
    window.title("FINAL YEAR PROJECT")
    window.configure(background='seagreen1')
    window.geometry('1500x1000')


    x_cord = 75;
    y_cord = 20;



    lbl = tk.Label(window, text="The product is   ",width=20  ,height=1  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=450-x_cord, y=500-y_cord)

    message3 = tk.Label(window, text="" ,bg="white"  ,fg="black"  ,width=30  ,height=1, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
    message3.place(x=750-x_cord, y=500-y_cord)




    message = tk.Label(window, text="DETECTION ,PRODUCT DETAILS & POPULARITY" ,bg="seagreen1"  ,fg="black"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
    message.place(x=170, y=20)




    lbl = tk.Label(window, text="NAME",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=200-y_cord)


    message1 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message1.place(x=250-x_cord, y=270-y_cord)

    lbl2 = tk.Label(window, text="CATEGORY",width=20  ,fg="black"  ,bg="seagreen1"    ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl2.place(x=600-x_cord, y=200-y_cord)



    message4 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message4.place(x=650-x_cord, y=270-y_cord)

    lbl3 = tk.Label(window, text="PRICE(₹)",width=20  ,fg="black"  ,bg="seagreen1"  ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl3.place(x=1000-x_cord, y=200-y_cord)

    message5 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message5.place(x=1060-x_cord, y=270-y_cord)

    lbl = tk.Label(window, text="Ratings(out of 5)",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=300-y_cord)


    message6 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message6.place(x=250-x_cord, y=370-y_cord)

    lbl = tk.Label(window, text="Net Worth",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=600-x_cord, y=300-y_cord)


    message7 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message7.place(x=650-x_cord, y=370-y_cord)


                 
                       

    def yolo():
        net = cv2.dnn.readNet("yolo-coco-data/yolov3_training_2000_L.weights", "yolo-coco-data/yolov3_testing_L.cfg")
        classes = []
        with open("yolo-coco-data/classes_L.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))



        cap =cv2.VideoCapture(0)

        co=1
        if(co==1):
            _, img=cap.read()
            height, width, channels = img.shape


            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)


            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                  
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
     
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            #print(indexes)
            counter = 1
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                cv2.imshow("Image", img)
                key=cv2.waitKey(0)
                if key==27:
                    break


                if len(indexes) > 0:
                    for i in indexes.flatten():
                        #print('Object {0}: {1}'.format(counter, classes[int(class_ids[i])]))
                        labels1=format(classes[int(class_ids[i])])
                        for j in labels1:
                            if(j.islower())==True:
                                a+=(j.upper())
                            else:
                                a=labels1
                        data=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck.csv');
                        csv_file=csv.reader(open('C:/Users/Home-PC/Desktop/final/dcck.csv'))
                        name=a
                        name1=""
                        for i in name:
                          if(i.islower())==True:
                              name1+=(i.upper())
                          else:
                              name1=name
                        for row in csv_file:
                          if name1==row[1]:
                              Name=row[1]
                              Catagory=row[2]
                              Price=row[3]
                              Ratings=row[4]
                              Net=row[5]
                              message1.configure(text=Name)
                              message4.configure(text=Catagory)
                              message5.configure(text=Price)
                              message6.configure(text=Ratings)
                              message7.configure(text=Net)
                              if row[2]=="FOOD":
                                  a=1
                              elif row[2]=="ELECTRONIC GADGET":
                                  a=2
                              elif row[2]=="STATIONARY":
                                  a=3
                              elif row[2]=="ACCESSORIES":
                                  a=4
                              elif row[2]=="PERSONAL CARE":
                                  a=5
                              else:
                                  a=6                                    
                              b=int(row[3])
                              c=float(row[4])
                              data1=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck 1.csv');
                              data1.drop(['ID'],axis=1,inplace=True)
                              data1.drop(['NAME'],axis=1,inplace=True)
                              X=data1[["CATEGORY","PRICE","RATING"]]
                              Y=data1.iloc[:,4]
                              from sklearn.model_selection import train_test_split
                              X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
                              from sklearn.linear_model import LogisticRegression
                              log=LogisticRegression()
                              log.fit(X_train,Y_train)                          
                              pred=log.predict(X_test)
                              pred1=log.predict([[a,b,c]])
                         
                              if(pred1==1):
                                     res= "famous in todays market"
                              elif(pred1==0.5):
                                     res=" average famous in todays market"
                              else:
                                     res="not famous in todays market"
                              message3.configure(text=res)

        co=co+1
        cap.release()     
        cv2.destroyAllWindows()

        
    def quit_window():
           window.destroy()




    trainImg = tk.Button(window, text="START TO DETECT AND PREDICT", command=yolo  ,fg="black"  ,bg="cyan3"  ,width=35  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=100)


    quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="cyan3"  ,width=10  ,height=2, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    quitWindow.place(x=1000, y=735-y_cord)

    window.mainloop()


def pred2():
    window =tk.Tk()
    window.title("FINAL YEAR PROJECT")
    window.configure(background='seagreen1')
    window.geometry('1500x1000')


    x_cord = 75;
    y_cord = 20;



    lbl = tk.Label(window, text="The product is   ",width=20  ,height=1  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=450-x_cord, y=500-y_cord)

    message3 = tk.Label(window, text="" ,bg="white"  ,fg="black"  ,width=30  ,height=1, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
    message3.place(x=750-x_cord, y=500-y_cord)




    message = tk.Label(window, text="DETECTION ,PRODUCT DETAILS & POPULARITY" ,bg="seagreen1"  ,fg="black"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
    message.place(x=170, y=20)




    lbl = tk.Label(window, text="NAME",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=200-y_cord)


    message1 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message1.place(x=250-x_cord, y=270-y_cord)

    lbl2 = tk.Label(window, text="CATEGORY",width=20  ,fg="black"  ,bg="seagreen1"    ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl2.place(x=600-x_cord, y=200-y_cord)



    message4 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message4.place(x=650-x_cord, y=270-y_cord)

    lbl3 = tk.Label(window, text="PRICE(₹)",width=20  ,fg="black"  ,bg="seagreen1"  ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl3.place(x=1000-x_cord, y=200-y_cord)

    message5 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message5.place(x=1060-x_cord, y=270-y_cord)

    lbl = tk.Label(window, text="Ratings(out of 5)",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=300-y_cord)


    message6 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message6.place(x=250-x_cord, y=370-y_cord)

    lbl = tk.Label(window, text="Net Worth",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=600-x_cord, y=300-y_cord)


    message7 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message7.place(x=650-x_cord, y=370-y_cord)


                 
                       

    def yolo():
        net = cv2.dnn.readNet("yolo-coco-data/yolov3_training_last_P.weights", "yolo-coco-data/yolov3_testing_P.cfg")
        classes = []
        with open("yolo-coco-data/classes1_P.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))



        cap =cv2.VideoCapture(0)

        co=1
        if(co==1):
            _, img=cap.read()
            height, width, channels = img.shape


            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)


            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                  
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
     
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            #print(indexes)
            counter = 1
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                cv2.imshow("Image", img)
                key=cv2.waitKey(0)
                if key==27:
                    break


                if len(indexes) > 0:
                    for i in indexes.flatten():
                        #print('Object {0}: {1}'.format(counter, classes[int(class_ids[i])]))
                        labels1=format(classes[int(class_ids[i])])
                        if labels1=="iphone_6":
                            a="IPHONE_6"
                        else:
                            a="REDMI NOTE 4"
                        data=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck.csv');
                        csv_file=csv.reader(open('C:/Users/Home-PC/Desktop/final/dcck.csv'))
                        name=a
                        name1=""
                        for i in name:
                          if(i.islower())==True:
                              name1+=(i.upper())
                          else:
                              name1=name
                        for row in csv_file:
                          if name1==row[1]:
                              Name=row[1]
                              Catagory=row[2]
                              Price=row[3]
                              Ratings=row[4]
                              Net=row[5]
                              message1.configure(text=Name)
                              message4.configure(text=Catagory)
                              message5.configure(text=Price)
                              message6.configure(text=Ratings)
                              message7.configure(text=Net)
                              if row[2]=="FOOD":
                                  a=1
                              elif row[2]=="ELECTRONIC GADGET":
                                  a=2
                              elif row[2]=="STATIONARY":
                                  a=3
                              elif row[2]=="ACCESSORIES":
                                  a=4
                              elif row[2]=="PERSONAL CARE":
                                  a=5
                              else:
                                  a=6                                    
                              b=int(row[3])
                              c=float(row[4])
                              data1=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck 1.csv');
                              data1.drop(['ID'],axis=1,inplace=True)
                              data1.drop(['NAME'],axis=1,inplace=True)
                              X=data1[["CATEGORY","PRICE","RATING"]]
                              Y=data1.iloc[:,4]
                              from sklearn.model_selection import train_test_split
                              X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
                              from sklearn.linear_model import LogisticRegression
                              log=LogisticRegression()
                              log.fit(X_train,Y_train)                          
                              pred=log.predict(X_test)
                              pred1=log.predict([[a,b,c]])
                         
                              if(pred1==1):
                                     res= "famous in todays market"
                              elif(pred1==0.5):
                                     res=" average famous in todays market"
                              else:
                                     res="not famous in todays market"
                              message3.configure(text=res)

        co=co+1
        cap.release()     
        cv2.destroyAllWindows()

        
    def quit_window():
           window.destroy()




    trainImg = tk.Button(window, text="START TO DETECT AND PREDICT", command=yolo  ,fg="black"  ,bg="cyan3"  ,width=35  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=100)


    quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="cyan3"  ,width=10  ,height=2, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    quitWindow.place(x=1000, y=735-y_cord)

    window.mainloop()


def pred3():
    window =tk.Tk()
    window.title("FINAL YEAR PROJECT")
    window.configure(background='seagreen1')
    window.geometry('1500x1000')


    x_cord = 75;
    y_cord = 20;



    lbl = tk.Label(window, text="The product is   ",width=20  ,height=1  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=450-x_cord, y=500-y_cord)

    message3 = tk.Label(window, text="" ,bg="white"  ,fg="black"  ,width=30  ,height=1, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
    message3.place(x=750-x_cord, y=500-y_cord)




    message = tk.Label(window, text="DETECTION ,PRODUCT DETAILS & POPULARITY" ,bg="seagreen1"  ,fg="black"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
    message.place(x=170, y=20)




    lbl = tk.Label(window, text="NAME",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=200-y_cord)


    message1 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message1.place(x=250-x_cord, y=270-y_cord)

    lbl2 = tk.Label(window, text="CATEGORY",width=20  ,fg="black"  ,bg="seagreen1"    ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl2.place(x=600-x_cord, y=200-y_cord)



    message4 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message4.place(x=650-x_cord, y=270-y_cord)

    lbl3 = tk.Label(window, text="PRICE(₹)",width=20  ,fg="black"  ,bg="seagreen1"  ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl3.place(x=1000-x_cord, y=200-y_cord)

    message5 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message5.place(x=1060-x_cord, y=270-y_cord)

    lbl = tk.Label(window, text="Ratings(out of 5)",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=300-y_cord)


    message6 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message6.place(x=250-x_cord, y=370-y_cord)

    lbl = tk.Label(window, text="Net Worth",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=600-x_cord, y=300-y_cord)


    message7 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message7.place(x=650-x_cord, y=370-y_cord)


                 
                       

    def yolo():
        net = cv2.dnn.readNet("yolo-coco-data/yolov3_training_last.weights", "yolo-coco-data/yolov3_testing.cfg")
        classes = []
        with open("yolo-coco-data/classes1.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))



        cap =cv2.VideoCapture(0)

        co=1
        if(co==1):
            _, img=cap.read()
            height, width, channels = img.shape


            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)


            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                  
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
     
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            #print(indexes)
            counter = 1
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                cv2.imshow("Image", img)
                key=cv2.waitKey(0)
                if key==27:
                    break


                if len(indexes) > 0:
                    for i in indexes.flatten():
                        #print('Object {0}: {1}'.format(counter, classes[int(class_ids[i])]))
                        labels1=format(classes[int(class_ids[i])])
                        for j in labels1:
                            if(j.islower())==True:
                                a+=(j.upper())
                            else:
                                a=labels1
                        data=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck.csv');
                        csv_file=csv.reader(open('C:/Users/Home-PC/Desktop/final/dcck.csv'))
                        name=a
                        name1=""
                        for i in name:
                          if(i.islower())==True:
                              name1+=(i.upper())
                          else:
                              name1=name
                        for row in csv_file:
                          if name1==row[1]:
                              Name=row[1]
                              Catagory=row[2]
                              Price=row[3]
                              Ratings=row[4]
                              Net=row[5]
                              message1.configure(text=Name)
                              message4.configure(text=Catagory)
                              message5.configure(text=Price)
                              message6.configure(text=Ratings)
                              message7.configure(text=Net)
                              if row[2]=="FOOD":
                                  a=1
                              elif row[2]=="ELECTRONIC GADGET":
                                  a=2
                              elif row[2]=="STATIONARY":
                                  a=3
                              elif row[2]=="ACCESSORIES":
                                  a=4
                              elif row[2]=="PERSONAL CARE":
                                  a=5
                              else:
                                  a=6                                    
                              b=int(row[3])
                              c=float(row[4])
                              data1=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck 1.csv');
                              data1.drop(['ID'],axis=1,inplace=True)
                              data1.drop(['NAME'],axis=1,inplace=True)
                              X=data1[["CATEGORY","PRICE","RATING"]]
                              Y=data1.iloc[:,4]
                              from sklearn.model_selection import train_test_split
                              X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
                              from sklearn.linear_model import LogisticRegression
                              log=LogisticRegression()
                              log.fit(X_train,Y_train)                          
                              pred=log.predict(X_test)
                              pred1=log.predict([[a,b,c]])
                         
                              if(pred1==1):
                                     res= "famous in todays market"
                              elif(pred1==0.5):
                                     res=" average famous in todays market"
                              else:
                                     res="not famous in todays market"
                              message3.configure(text=res)

        co=co+1
        cap.release()     
        cv2.destroyAllWindows()

        
    def quit_window():
           window.destroy()




    trainImg = tk.Button(window, text="START TO DETECT AND PREDICT", command=yolo  ,fg="black"  ,bg="cyan3"  ,width=35  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=100)


    quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="cyan3"  ,width=10  ,height=2, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    quitWindow.place(x=1000, y=735-y_cord)

    window.mainloop()


def pred4():
    window =tk.Tk()
    window.title("FINAL YEAR PROJECT")
    window.configure(background='seagreen1')
    window.geometry('1500x1000')


    x_cord = 75;
    y_cord = 20;



    lbl = tk.Label(window, text="The product is   ",width=20  ,height=1  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=450-x_cord, y=500-y_cord)

    message3 = tk.Label(window, text="" ,bg="white"  ,fg="black"  ,width=30  ,height=1, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
    message3.place(x=750-x_cord, y=500-y_cord)




    message = tk.Label(window, text="DETECTION ,PRODUCT DETAILS & POPULARITY" ,bg="seagreen1"  ,fg="black"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
    message.place(x=170, y=20)




    lbl = tk.Label(window, text="NAME",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=200-y_cord)


    message1 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message1.place(x=250-x_cord, y=270-y_cord)

    lbl2 = tk.Label(window, text="CATEGORY",width=20  ,fg="black"  ,bg="seagreen1"    ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl2.place(x=600-x_cord, y=200-y_cord)



    message4 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message4.place(x=650-x_cord, y=270-y_cord)

    lbl3 = tk.Label(window, text="PRICE(₹)",width=20  ,fg="black"  ,bg="seagreen1"  ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl3.place(x=1000-x_cord, y=200-y_cord)

    message5 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message5.place(x=1060-x_cord, y=270-y_cord)

    lbl = tk.Label(window, text="Ratings(out of 5)",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=300-y_cord)


    message6 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message6.place(x=250-x_cord, y=370-y_cord)

    lbl = tk.Label(window, text="Net Worth",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=600-x_cord, y=300-y_cord)


    message7 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message7.place(x=650-x_cord, y=370-y_cord)


                 
                       

    def yolo():
        net = cv2.dnn.readNet("yolo-coco-data/yolov3_training_2000_U.weights", "yolo-coco-data/yolov3_testing_U.cfg")
        classes = []
        with open("yolo-coco-data/classes_U.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))



        cap =cv2.VideoCapture(0)

        co=1
        if(co==1):
            _, img=cap.read()
            height, width, channels = img.shape


            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)


            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                  
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
     
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            #print(indexes)
            counter = 1
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                cv2.imshow("Image", img)
                key=cv2.waitKey(0)
                if key==27:
                    break


                if len(indexes) > 0:
                    for i in indexes.flatten():
                        #print('Object {0}: {1}'.format(counter, classes[int(class_ids[i])]))
                        labels1=format(classes[int(class_ids[i])])
                        for j in labels1:
                            if(j.islower())==True:
                                a+=(j.upper())
                            else:
                                a=labels1
                        data=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck.csv');
                        csv_file=csv.reader(open('C:/Users/Home-PC/Desktop/final/dcck.csv'))
                        name=a
                        name1=""
                        for i in name:
                          if(i.islower())==True:
                              name1+=(i.upper())
                          else:
                              name1=name
                        for row in csv_file:
                          if name1==row[1]:
                              Name=row[1]
                              Catagory=row[2]
                              Price=row[3]
                              Ratings=row[4]
                              Net=row[5]
                              message1.configure(text=Name)
                              message4.configure(text=Catagory)
                              message5.configure(text=Price)
                              message6.configure(text=Ratings)
                              message7.configure(text=Net)
                              if row[2]=="FOOD":
                                  a=1
                              elif row[2]=="ELECTRONIC GADGET":
                                  a=2
                              elif row[2]=="STATIONARY":
                                  a=3
                              elif row[2]=="ACCESSORIES":
                                  a=4
                              elif row[2]=="PERSONAL CARE":
                                  a=5
                              else:
                                  a=6                                    
                              b=int(row[3])
                              c=float(row[4])
                              data1=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck 1.csv');
                              data1.drop(['ID'],axis=1,inplace=True)
                              data1.drop(['NAME'],axis=1,inplace=True)
                              X=data1[["CATEGORY","PRICE","RATING"]]
                              Y=data1.iloc[:,4]
                              from sklearn.model_selection import train_test_split
                              X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
                              from sklearn.linear_model import LogisticRegression
                              log=LogisticRegression()
                              log.fit(X_train,Y_train)                          
                              pred=log.predict(X_test)
                              pred1=log.predict([[a,b,c]])
                         
                              if(pred1==1):
                                     res= "famous in todays market"
                              elif(pred1==0.5):
                                     res=" average famous in todays market"
                              else:
                                     res="not famous in todays market"
                              message3.configure(text=res)

        co=co+1
        cap.release()     
        cv2.destroyAllWindows()

        
    def quit_window():
           window.destroy()




    trainImg = tk.Button(window, text="START TO DETECT AND PREDICT", command=yolo  ,fg="black"  ,bg="cyan3"  ,width=35  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=100)


    quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="cyan3"  ,width=10  ,height=2, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    quitWindow.place(x=1000, y=735-y_cord)

    window.mainloop()


def pred5():
    window =tk.Tk()
    window.title("FINAL YEAR PROJECT")
    window.configure(background='seagreen1')
    window.geometry('1500x1000')


    x_cord = 75;
    y_cord = 20;



    lbl = tk.Label(window, text="The product is   ",width=20  ,height=1  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=450-x_cord, y=500-y_cord)

    message3 = tk.Label(window, text="" ,bg="white"  ,fg="black"  ,width=30  ,height=1, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
    message3.place(x=750-x_cord, y=500-y_cord)




    message = tk.Label(window, text="DETECTION ,PRODUCT DETAILS & POPULARITY" ,bg="seagreen1"  ,fg="black"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
    message.place(x=170, y=20)




    lbl = tk.Label(window, text="NAME",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=200-y_cord)


    message1 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message1.place(x=250-x_cord, y=270-y_cord)

    lbl2 = tk.Label(window, text="CATEGORY",width=20  ,fg="black"  ,bg="seagreen1"    ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl2.place(x=600-x_cord, y=200-y_cord)



    message4 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message4.place(x=650-x_cord, y=270-y_cord)

    lbl3 = tk.Label(window, text="PRICE(₹)",width=20  ,fg="black"  ,bg="seagreen1"  ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
    lbl3.place(x=1000-x_cord, y=200-y_cord)

    message5 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message5.place(x=1060-x_cord, y=270-y_cord)

    lbl = tk.Label(window, text="Ratings(out of 5)",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=200-x_cord, y=300-y_cord)


    message6 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message6.place(x=250-x_cord, y=370-y_cord)

    lbl = tk.Label(window, text="Net Worth",width=20  ,height=2  ,fg="black"  ,bg="seagreen1" ,font=('Times New Roman', 25, ' bold ') ) 
    lbl.place(x=600-x_cord, y=300-y_cord)


    message7 = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=25  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
    message7.place(x=650-x_cord, y=370-y_cord)


                 
                       

    def yolo():
        net = cv2.dnn.readNet("yolo-coco-data/yolov3_training_last_D.weights", "yolo-coco-data/yolov3_testing_D.cfg")
        classes = []
        with open("yolo-coco-data/classes_D.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))



        cap =cv2.VideoCapture(0)

        co=1
        if(co==1):
            _, img=cap.read()
            height, width, channels = img.shape


            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)


            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                  
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
     
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            #print(indexes)
            counter = 1
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                cv2.imshow("Image", img)
                key=cv2.waitKey(0)
                if key==27:
                    break


                if len(indexes) > 0:
                    for i in indexes.flatten():
                        #print('Object {0}: {1}'.format(counter, classes[int(class_ids[i])]))
                        labels1=format(classes[int(class_ids[i])])
                        for j in labels1:
                            if(j.islower())==True:
                                a+=(j.upper())
                            else:
                                a=labels1
                        data=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck.csv');
                        csv_file=csv.reader(open('C:/Users/Home-PC/Desktop/final/dcck.csv'))
                        name=a
                        name1=""
                        for i in name:
                          if(i.islower())==True:
                              name1+=(i.upper())
                          else:
                              name1=name
                        for row in csv_file:
                          if name1==row[1]:
                              Name=row[1]
                              Catagory=row[2]
                              Price=row[3]
                              Ratings=row[4]
                              Net=row[5]
                              message1.configure(text=Name)
                              message4.configure(text=Catagory)
                              message5.configure(text=Price)
                              message6.configure(text=Ratings)
                              message7.configure(text=Net)
                              if row[2]=="FOOD":
                                  a=1
                              elif row[2]=="ELECTRONIC GADGET":
                                  a=2
                              elif row[2]=="STATIONARY":
                                  a=3
                              elif row[2]=="ACCESSORIES":
                                  a=4
                              elif row[2]=="PERSONAL CARE":
                                  a=5
                              else:
                                  a=6                                    
                              b=int(row[3])
                              c=float(row[4])
                              data1=pd.read_csv('C:/Users/Home-PC/Desktop/final/dcck 1.csv');
                              data1.drop(['ID'],axis=1,inplace=True)
                              data1.drop(['NAME'],axis=1,inplace=True)
                              X=data1[["CATEGORY","PRICE","RATING"]]
                              Y=data1.iloc[:,4]
                              from sklearn.model_selection import train_test_split
                              X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
                              from sklearn.linear_model import LogisticRegression
                              log=LogisticRegression()
                              log.fit(X_train,Y_train)                          
                              pred=log.predict(X_test)
                              pred1=log.predict([[a,b,c]])
                         
                              if(pred1==1):
                                     res= "famous in todays market"
                              elif(pred1==0.5):
                                     res=" average famous in todays market"
                              else:
                                     res="not famous in todays market"
                              message3.configure(text=res)

        co=co+1
        cap.release()     
        cv2.destroyAllWindows()

        
    def quit_window():
           window.destroy()




    trainImg = tk.Button(window, text="START TO DETECT AND PREDICT", command=yolo  ,fg="black"  ,bg="cyan3"  ,width=35  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=100)


    quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="cyan3"  ,width=10  ,height=2, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    quitWindow.place(x=1000, y=735-y_cord)

    window.mainloop()



def cat():
    window =tk.Tk()
    window.title("FINAL YEAR PROJECT")
    window.configure(background='seagreen1')
    window.geometry('1500x1000')


    x_cord = 75;
    y_cord = 20;

    def quit_window():
           window.destroy()

    
    message = tk.Label(window, text="SELECT PRODUCT CATEGORY" ,bg="seagreen1"  ,fg="black"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
    message.place(x=170, y=20)

    trainImg = tk.Button(window, text="FOOD", command=pred1  ,fg="black"  ,bg="cyan3"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=150)

    trainImg = tk.Button(window, text="ELECTRONIC GADGET", command=pred2  ,fg="black"  ,bg="cyan3"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=250)
    
    #trainImg = tk.Button(window, text="STATIONARY", command=pred3  ,fg="black"  ,bg="cyan3"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    #77777777trainImg.place(x=550, y=350)
    
    trainImg = tk.Button(window, text="DETERGENT", command=pred5  ,fg="black"  ,bg="cyan3"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=450)
    
    trainImg = tk.Button(window, text="Washing", command=pred4  ,fg="black"  ,bg="cyan3"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=550)


    quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="cyan3"  ,width=10  ,height=2, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    quitWindow.place(x=1000, y=735-y_cord)

    window.mainloop()


    

class start:
    def __init__(self):
        window =Tk()
        window=Canvas(window,width=1500,height=1000,background='seagreen1')
        window.pack()
        image = PhotoImage(file = 'C:\\Users\\Home-PC\\Desktop\\final\\images\\5.png')
        window.create_image(400,200, anchor = NW, image = image)
        message = tk.Label(window, text="IDENTIFYING AND  DEVELOPING PREDICTIVE MODELS" ,bg="seagreen1"  ,fg="black"  ,width=50  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
        message.place(x=10, y=20)
        message = tk.Label(window, text="USING MACHINE LEARNING ALGORITHMS FOR ENHANCING SALES " ,bg="seagreen1"  ,fg="black"  ,width=60  ,height=1,font=('Times New Roman', 25, 'bold underline')) 
        message.place(x=100, y=80)
        imagetest = PhotoImage(file='C:\\Users\\Home-PC\\Desktop\\final\\images\\2.png')
        window.create_image(250, 400, image=imagetest)
        imagetest1 = PhotoImage(file='C:\\Users\\Home-PC\\Desktop\\final\\images\\1.png')
        window.create_image(1200, 400, image=imagetest1)
        x_cord = 75;
        y_cord = 20;

        def quit_window():
            MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the application',icon = 'warning')
            if MsgBox == 'yes':
                tk.messagebox.showinfo("Greetings", "Thank You very much for using our software. Have a nice day ahead!!")
                window.destroy()




        quitWindow = tk.Button(window, text="Prediction", command=cat  ,fg="black"  ,bg="seagreen",width=15  ,height=2, activebackground = "BLUE" ,font=('Times New Roman', 15, ' bold '))
        quitWindow.place(x=155, y=500)        

        quitWindow = tk.Button(window, text="Damage", command=damage  ,fg="black"  ,bg="seagreen",width=15  ,height=2, activebackground = "BLUE" ,font=('Times New Roman', 15, ' bold '))
        quitWindow.place(x=1100, y=500)

        quitWindow = tk.Button(window, text="QUIT", command=quit_window  ,fg="black"  ,bg="seagreen2"  ,width=10  ,height=2, activebackground = "BLUE" ,font=('Times New Roman', 15, ' bold '))
        quitWindow.place(x=800, y=735-y_cord)
        window.mainloop()



if(1):
     GUUEST=start()




