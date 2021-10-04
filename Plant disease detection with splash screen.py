import sys
import os
import platform
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PySide2.QtWidgets import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from PIL import Image, ImageTk
import os
import cv2
from tkinter import *
from PIL import ImageTk, Image
import _tkinter

win=tk.Tk()
frm = Frame(win)
frm.pack(side=BOTTOM, padx=15, pady=15)
lbl = Label(win)
lbl.pack()
lbl1=Label(win)

## ==> SPLASH SCREEN
from ui_splash_screen import Ui_SplashScreen

## ==> MAIN WINDOW
from ui_main import Ui_MainWindow

## ==> GLOBALS
counter = 0

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        def b1_click():
            global path2
            try:
                json_file = open('model1.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                loaded_model.load_weights("model1.h5")
                print("Loaded model from disk")
                label=["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
                       "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
                       "Corn_(maize)___Healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
                       "Grape___Esca_(Black_Measles)","Grape___Healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                       "Potato___Early_blight","Potato___Healthy","Potato___Late_blight","Tomato___Bacterial_spot",
                       "Tomato___Early_blight","Tomato___Healthy","Tomato___Late_blight","Tomato___Leaf_Mold",
                       "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
                       "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus"]

                path2=filedialog.askopenfilename(initialdir=os.getcwd(), title='Select Image File', filetypes=(("JPG File", "*.jpg"), ("PNG File", "*.png"), ("All Files", "*.*")))
                displayimg = Image.open(path2)
                displayimg.thumbnail((250,250))
                displayimg=ImageTk.PhotoImage(displayimg)
                lbl.configure(image =displayimg)
                lbl.image =displayimg
                print(path2)

                test_image = image.load_img(path2, target_size = (128, 128))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                result = loaded_model.predict(test_image)

                fresult=np.max(result)
                label2=label[result.argmax()]
                print(label2)
                lbl1.configure(text=label2)

                win.mainloop()


            except IOError:
                pass

        def exit():
            win.destroy()


        label1 = Label(win, text="GUI For Leaf Disease Detection using OPENCV", fg ='blue')
        label1.pack()

        b1=tk.Button(frm, text="Upload Image",width=25, height=3,font=('bold',10), activebackground='green',activeforeground='white',command=b1_click)
        b1.pack(side=tk.LEFT)

        b2=tk.Button(frm, text="Exit",width=25, height=3,font=('bold',10), activebackground='Red',activeforeground='white',command=exit)
        b2.pack(side=tk.LEFT, padx=10)

        lbl =Label(win, text="Input Image", fg ='Violet')
        lbl.pack()
        lbl1= Label(win, text="Result", fg="Green")
        lbl1.pack()

        win.geometry("400x400")
        win.title("Plant Disease Detection")
        win.bind("<Return>",b1_click)
        win.mainloop()


# SPLASH SCREEN
class SplashScreen(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_SplashScreen()
        self.ui.setupUi(self)

    

        ## REMOVE TITLE BAR
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)


        ## DROP SHADOW EFFECT
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.ui.dropShadowFrame.setGraphicsEffect(self.shadow)

        ## QTIMER ==> START
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        # TIMER IN MILLISECONDS
        self.timer.start(100)

        # Initial Text
        self.ui.label_description.setText("<strong>WELCOME</strong> TO OUR APPLICATION")

        # Change Texts
        QtCore.QTimer.singleShot(1500, lambda: self.ui.label_description.setText("<strong>LOADING</strong> DATABASE"))
        QtCore.QTimer.singleShot(3000, lambda: self.ui.label_description.setText("<strong>LOADING</strong> USER INTERFACE"))
        QtCore.QTimer.singleShot(4500, lambda: self.ui.label_description.setText("<strong>LOADING</strong> ENVIRONEMENT"))
        QtCore.QTimer.singleShot(5500, lambda: self.ui.label_description.setText("<strong>LOADING</strong> VALUES"))
        QtCore.QTimer.singleShot(6000, lambda: self.ui.label_description.setText("<strong>LOADING</strong> SIMPLE GUI"))
        QtCore.QTimer.singleShot(7000, lambda: self.ui.label_description.setText("<strong>LOADING</strong> MODELS"))


        ## SHOW ==> MAIN WINDOW
        ########################################################################
        self.show()
        ## ==> END ##

    ## ==> APP FUNCTIONS
    ########################################################################
    def progress(self):
        global counter

        # SET VALUE TO PROGRESS BAR
        self.ui.progressBar.setValue(counter)

        # CLOSE SPLASH SCREE AND OPEN APP
        if counter > 100:
            # STOP TIMER
            self.timer.stop()

            # SHOW MAIN WINDOW
            self.main = MainWindow()
            self.main.close()

            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        counter += 1




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SplashScreen()
    sys.exit(app.exec_())
