import tensorflow as tf             #ML
import tkinter as tk                #GUI
import numpy as np                  #matrices

#drawing on draw pad
def on_drag(event):
    #getting mouse coordinates
    x, y = window.winfo_pointerxy()

    #if in draw pad area
    if x > 42 and x < 518 and y > 100 and y < 576:
        #get the position of the mouse relative to canvas grid
        sqrX = (int)((x - 42) / 17)
        sqrY = (int)((y - 100) / 17)

        #if the pixel is not activated
        if int(pixel_value[sqrX +  (28 * sqrY)]) == 0:
            #activate pixel
            pixel_value[sqrX +  (28 * sqrY)] = 1
            #color grid
            canvas.create_rectangle(42 + ((sqrX * 17)), 42 + ((sqrY * 17)), 42 + ((sqrX * 17)) + 17, 42 + ((sqrY * 17)) + 17, fill="white")

        #pass pixel activation into network
        prediction = model.predict(tf.convert_to_tensor(pixel_value.reshape(1, 28, 28, 1), dtype=tf.float32))
        
        #update confidences
        for i in range(0, 10):
            var[i].set("Number " + str(i) + ": " + str(int(prediction[0][i] * 10000)/100) + "%")

#creating drawing canvas
def zero_out():
    #zeroing out the pixel values
    global pixel_value
    pixel_value = np.zeros(784)

    #for all canvas pixels in the x
    for sqrX in range(0, 28):
        # for all canvas pixels in the y
        for sqrY in range(0, 28):
            #drawing canvas pixels
            canvas.create_rectangle(42 + ((sqrX * 17)), 42 + ((sqrY * 17)), 42 + ((sqrX * 17)) + 17, 42 + ((sqrY * 17)) + 17, fill="grey12")

    #for all grid placements
    for i in range(42, 535, 17):
        #drawing grid lines
        canvas.create_line(i, 42, i, 518, fill="grey39")
        canvas.create_line(42, i, 518, i, fill="grey39")

    for i in range(0, 10):
            var[i].set("Number " + str(i) + ": 0.0%")


#--------------------------------------------------------------------------------
#Model

#loading dataset
mnist = tf.keras.datasets.mnist

#split dataset into training and testing data
(training_image, training_label), (testing_image, testing_label) = mnist.load_data()

#loading previously saved model
model = tf.keras.models.load_model('handwritten_digits_1.0')

# evaluating model based on test data
loss, accuracy = model.evaluate(testing_image, testing_label)


#--------------------------------------------------------------------------------
#GUI - Draw Pad

#pop-up window
window = tk.Tk()
#windom name
window.title("Digit Recognition")

#draw pad frame
frameCanvas = tk.Frame(master=window, width=560, height=560)
frameCanvas.pack(fill=tk.BOTH, side=tk.LEFT)

#prediction confedence frame
framePrediction = tk.Frame(master=window, width=300, height=560, bg="grey20")
framePrediction.pack(fill=tk.BOTH, side=tk.LEFT)

#creating confidence label text
var = []
for i in range(10):
    var.append(tk.StringVar())

#spacer label
labelSpacer0 = tk.Label(framePrediction, width=25, height=1, bg="grey20")
labelSpacer0.pack()

#displaying loss label
labelLoss = tk.Label(framePrediction, text="Loss: " + str(int(loss*100)/100), width=25, height=1, bg="grey20", fg="white")
labelLoss.pack()

#displaying accuracy label
labelAcc = tk.Label(framePrediction, text="Accuracy: " + str(int(accuracy*10000)/100), width=25, height=1, bg="grey20", fg="white")
labelAcc.pack()

#creating confidence labels
label = []
for i in range(10):
    label.append(tk.Label(framePrediction, textvariable=var[i], width=25, height=2, bg="grey20", fg="white", justify="left", anchor="w"))
    label[i].place(x=65, y=120+(i*30))
    # label[i].pack()

#creating clear button
clearBut = tk.Button(framePrediction, text="Clear", highlightbackground="grey20", command=zero_out).place(x=80, y=450)

#draw pad
canvas = tk.Canvas(frameCanvas, bg="grey12", width=560, height=560)
canvas.pack()

#creating the grid and the pixel array
zero_out()

#28x28 box coordinate array
square = np.zeros((28, 28, 2))

#numpy starting index
x = 0

#coordinate array initialization
#for all horizontal squares
for i in range(42, 535, 17):
    #numpy secondary starting index
    y = 0
    
    #for all vertical squares
    for j in range(42, 535, 17):
        #declaring grind coordinates
        square[x][y] = i, j

        if y < 27:
            y += 1

    if x < 27:
        x += 1

#when mouse is clicked and dragged - drawing
window.bind("<B1-Motion>", on_drag)
# window.bind("<Return>", displayArray)

#updating screen
window.mainloop()