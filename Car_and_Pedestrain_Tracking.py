import cv2



#Video of cars on road
#video = cv2.VideoCapture('Tesla_Car_Video.mp4')
video = cv2.VideoCapture('Pedestrian_Video.mp4')

#Create Car and Pedestrian Classifier from pre-trained classifiers
car_tracker = cv2.CascadeClassifier('cars.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

#For video to run forever until car stops
while True:
    #Read the current frame
    (read_successful, frame) = video.read() #reads single frame, and continues reading frames on a loop

    # Safe Coding
    if read_successful:
        #Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    #Detects Cars and pedestrians at any Scale
    #cars = car_tracker.detectMultiScale(grayscaled_frame) 
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw rectangles around the cars detected
    #for (x,y,h,w) in cars:
    #    cv2.rectangle(frame,(x,y), (x+w,y+h), (0,0,255), 2) #0,0,255 is color of rectangle, BGR so red in this case and 2 is the thickness of rectangle

    #Draw yellow rectangles around the pedestrians detected
    for (x,y,h,w) in pedestrians:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,255), 2) 

    #Display the image/video with the faces spotted
    cv2.imshow('Car Detector',frame) #grayscaled is faster as it processes faster in python

    key = cv2.waitKey(1) #Makes sure the picture/video doesn't autoclose and waits for keypress 
    #if i use waitKey() it is stuck on one frame. Can be used for image.

    # Stop if Q or q is pressed
    if key == 81 or key == 113:
        break


"""
#This commented block is for image and only these lines vary. the rest remain the same for the video

#Image of Car in road
img_file = "Car_Image.jpg"

#create opencv image 
img = cv2.imread(img_file) #reads pixels from the image into multi dimensional array a.k.a img in this case

#convert image to grayscale (needed for haar cascade) - makes it much faster as well
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(cars) #first 2 numbers are top left side of rectangle, the following 2 numbers are the width and the height 



"""
 
print("Code Completed")