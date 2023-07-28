import cv2
import numpy as np
import face_recognition
import os

path = 'pplface'
imgs = []
names = []
oglist = os.listdir(path)
print(oglist)
for i in oglist:
    img = cv2.imread(f'{path}/{i}')
    imgs.append(img)
    names.append(os.path.splitext(i)[0])

print(names)

def encoder(imgs):
    encoded = []
    for i in imgs:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encoded.append(face_recognition.face_encodings(i)[0])
    return encoded

RecordEncoded = encoder(imgs)

print(len(RecordEncoded))

vid = cv2.VideoCapture(0)

while True:
    success, vid_img = vid.read()

    LRes = cv2.resize(vid_img,(300,400))
    LRes = cv2.cvtColor(LRes, cv2.COLOR_BGR2RGB)

    CFrame = face_recognition.face_locations(LRes)
    ECFrame = face_recognition.face_encodings(LRes, CFrame)

    for EFace, LFace in zip(ECFrame, CFrame):
        same = face_recognition.compare_faces(RecordEncoded, EFace)
        DFace = face_recognition.face_distance(RecordEncoded, EFace)
        sameI = np.argmin(DFace)

        if same[sameI]:
            name = names[sameI]
            print(name)
            y1, x2, y2, x1 = LFace
            cv2.rectangle(vid_img,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.rectangle(vid_img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(vid_img,name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,0.5, (255,255,255),2)


    cv2.imshow("Video", vid_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
