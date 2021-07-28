import cv2
import clx.xms
import requests



def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    name = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)

        # Predicting the id of the user
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])

        # Check for id of user and label the rectangle accordingly
        if id==1:
            name.append('Anwesha')
            cv2.putText(img, "Anwesha", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif id==2:
            name.append('Ayushman')
            cv2.putText(img, "Ayushman", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            name.append('Unknown')
            cv2.putText(img, "Unknown", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords, name

# Method to recognize the person
def recognize(img, clf, faceCascade):
    coords, name = draw_boundary(img, faceCascade, 1.1, 10, (255, 0, 0), "Face", clf)
    return img, name


# Loading classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

counter = 0

# Loading custom classifier to recognize
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
cap = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    _, img = cap.read()

    # Call method we defined above
    img, name = recognize(img, clf, faceCascade)

    client = clx.xms.Client(service_plan_id='9805c602cfb247748e0207edc6669765', token='efd11ba2d4f248a9a25d478968241404')

    create = clx.xms.api.MtBatchTextSmsCreate()
    create.sender = '447537455281'
    create.recipients = {'917735740041'}

    delimiter = ', '
    naming = delimiter.join(name)
    create.body = naming + ' is at your door'


    for face in img:
        if(face.any() and counter ==0):
            try:
               batch = client.create_batch(create)
            except (requests.exceptions.RequestException, clx.xms.exceptions.ApiException) as ex:
                print('Failed to communicate with XMS: %s' % str(ex))

        counter = 1

    # Writing processed image in a new window
    cv2.imshow("Face detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
cap.release()
# Destroying output window
cv2.destroyAllWindows()
