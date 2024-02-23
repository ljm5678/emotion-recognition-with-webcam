import cv2
import face_recognition
import dlib
import numpy as np
from keras.preprocessing.image import img_to_array
import tensorflow.keras.models as models

curr_frame = True
model = models.load_model('./output/checkpoints./epochs_20.keras')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if curr_frame:
        curr_frame = False
        rgb_image = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_loc = face_recognition.face_locations(rgb_image)

        for (top, right, bottom, left) in face_loc:
            top = top * 4
            bottom = bottom * 4 + 50
            right = right * 4
            left = left * 4
            if left < 0 or right < 0 or bottom < 0 or top < 0:
                break
            cropped_image = frame[top:bottom, left:right]
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (48, 48))
            image = resized_image.astype(np.float32) / 255.0
            image1 = img_to_array(image)
            image2 = np.expand_dims(image, axis=0)
            model_prediction = model.predict(image2)

            emotion = np.argmax(model_prediction)
            if emotion == 0:
                text = 'Angry'
            elif emotion == 5:
                text = 'Neutral'
            elif emotion == 2:
                text = 'Happy'
            elif emotion == 1:
                text = 'Fear'
            elif emotion == 4:
                text = 'Surprise'
            else:
                text = 'Sad'
            print (model_prediction)
            percentages = [
                "{:.2f}% Angry".format(model_prediction[0][0] * 100),
                "{:.2f}% Fear".format(model_prediction[0][1] * 100),
                "{:.2f}% Happy".format(model_prediction[0][2] * 100),
                "{:.2f}% Sad".format(model_prediction[0][3] * 100),
                "{:.2f}% Surprise".format(model_prediction[0][4] * 100),
                "{:.2f}% Neutral".format(model_prediction[0][5] * 100)
            ]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            curr = 0
            each = (bottom - top) // 6
            for i in range(6):
                cv2.rectangle(frame, (right, top + curr), (right + int(model_prediction[0][i] * 200), (top + curr + each)), (0, 0, 255), 2)
                cv2.putText(frame, percentages[i], (right + 6, top + curr + each // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                curr = curr + each




            cv2.putText(frame, text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


        cv2.imshow('Face and Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        curr_frame = True

video_capture.release(0)
cv2.destroyAllWindows()
