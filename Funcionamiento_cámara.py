import cv2
import face_recognition

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    ret,frame = cap.read()
    if ret == False: break
    frame = cv2.flip(frame,1)

    face_locations = face_recognition.face_locations(frame)
    if face_locations != []:
        for face in face_locations:
            cv2.rectangle(frame,(face[3],face[0]),(face[1],face[2]),(0,255,0),2)

    cv2.imshow("",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
