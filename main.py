import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('sentiment_model.h5')
sentiment_dict = {1:"anger",2:"contempt",3:"digust",4:"fear",5:"happy",6:"neutral",7:"sad",8:"surprise"}
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    img = cv2.resize(frame, (64, 64)) 
    img = img / 255.0 
    img = np.expand_dims(img, axis=0) 

    predictions = model.predict(img)
    sentiment_class = np.argmax(predictions)

    cv2.putText(frame, f'Sentiment: {sentiment_dict[sentiment_class]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()