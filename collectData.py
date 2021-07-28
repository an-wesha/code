import cv2

def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)

#drawing boundary
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):

    #convert colored image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    #define coordinates of bounary rectangle
    coords = []

    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, text, (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]

    return coords


#detecting face
def detect(img, faceCascade, img_id):
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255,0,0), "Face")

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        user_id = 3
        generate_dataset(roi_img, user_id, img_id)

    return img

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img_id = 0

cap = cv2.VideoCapture(0)

#video capture
while(cap.isOpened()):
    _, img = cap.read()

    img = detect(img, faceCascade, img_id)

    cv2.imshow('Frame', img)        #display
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):    #q is the key that we will press to exit
        break      #on pressing q we break out of the loop, videocap closes


cap.release()       #release the variables
cv2.destroyAllWindows()     #delete all frames
