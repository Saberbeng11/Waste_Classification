import cv2
import cvzone
import numpy as np
from keras.models import load_model
import serial.tools.list_ports

serialInt = serial.Serial()
serialInt.baudrate = 9600
serialInt.port = 'COM5'
serialInt.open()

# Load model:
my_model = load_model('Resources/Model/Garbage_Classifier.h5')

# Define Classes:
class_name = ["battery", "biological", "cardboard", "clothes", "glass" ,"metal", "paper", "plastic", "shoes", "trash"]

# Build Background:
arrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)

# Open Camera:
cap = cv2.VideoCapture(0)
while(True):
    ret, image_org = cap.read()
    if not ret:
        continue
    imgBackground = cv2.imread('Resources/Garbage Classification.png')
    cv2.rectangle(image_org, (0,0), (244,244), (0,0,255), 2)
    image = image_org.copy()
    image = image[0:224,0:224]
    image = image.astype('float')*1./255
    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0], axis=0))

    # Show text:
    if (np.max(predict) >= 0.7):
        # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    # GUI
    imgBackground[230:230 + 480, 14:14 + 640] = image_org

    if (np.max(predict) >= 0.7):
        i = np.argmax(predict)
        cv2.putText(imgBackground, class_name[i], (915, 257),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(imgBackground, str(round(np.max(predict[0], axis=0), 2)), (915, 288),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        if (i==1):
            cv2.putText(imgBackground, "Rac thai huu co", (915, 319),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            imgBackground = cvzone.overlayPNG(imgBackground, arrow, (729, 431))
            serialInt.write(b'm')
        elif(i==2 or i==5 or i==6 or i==7):
            cv2.putText(imgBackground, "Rac thai tai che", (915, 319),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            imgBackground = cvzone.overlayPNG(imgBackground, arrow, (1194, 431))
            serialInt.write(b'n')
        elif (i == 0):
            cv2.putText(imgBackground, "Rac thai doc hai", (915, 319),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            imgBackground = cvzone.overlayPNG(imgBackground, arrow, (884, 431))
            serialInt.write(b'o')
        else:
            cv2.putText(imgBackground, "Rac thai thuong", (915, 319),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            imgBackground = cvzone.overlayPNG(imgBackground, arrow, (1038, 431))
            serialInt.write(b'p')
    else:
        serialInt.write(b'k')


    cv2.imshow("Trí tuệ nhân tạo", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
serialInt.write(b'k')
# When everything done, release the capture
serialInt.close()
cap.release()
cv2.destroyAllWindows()
