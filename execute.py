import os
import os.path as osp
from multiprocessing.pool import ThreadPool
import cv2
from mpfd import *
#facedt=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
import numba as numba

facedt=mpfd()
def submosaic(bags):
    frame,i,y1,y2,boxsize=bags
    for j in range(y1,y2-boxsize,boxsize):
        rect=[i,j,boxsize,boxsize]
        color=frame[j+boxsize//2][i+boxsize//2].tolist()
        lu=(rect[0],rect[1])
        rd=(lu[0]+boxsize-1,lu[1]+boxsize-1)
        cv2.rectangle(frame,lu,rd,color,-1)
def mosaic(frame,x1,y1,x2,y2,boxsize=18):
    #for i in range(x1,x2-boxsize,boxsize):
        #submosaic(frame,i,y1,y2,boxsize)
    plist=[]
    for i in range(x1,x2-boxsize,boxsize):
        plist.append((frame,i,y1,y2,boxsize))
    pool=ThreadPool()
    pool.map(submosaic,plist)
    pool.close()
    pool.join()
def execute(vf=0):
    r=cv2.VideoCapture(vf)
    props={"fps":int(round(r.get(cv2.CAP_PROP_FPS))),"fcc":int(round(r.get(cv2.CAP_PROP_FOURCC))),
        "wh":(int(r.get(cv2.CAP_PROP_FRAME_WIDTH)),int(r.get(cv2.CAP_PROP_FRAME_HEIGHT)))}
    #timegap=1000//30
    #cv2.namedWindow("raw")
    #cv2.namedWindow("mosaic")
    frames=[]
    d=0
    while True:
        print(f"process frame {d}")
        #cv2.waitKey(1)
        d+=1
        st,frame=r.read()
        if not st:
            break
        #cv2.imshow("raw",frame)
        faces=facedt(frame)
        for facebox in faces:
            #rfb=facebox.rect
            rfb=facebox
            x1,y1,x2,y2=rfb.left(),rfb.top(),rfb.right(),rfb.bottom()
            mosaic(frame,x1,y1,x2,y2)
        #cv2.imshow("mosaic",frame)
        frames.append(frame)
    return (frames,props)
def master(realtime=True):
    fourcc=cv2.VideoWriter_fourcc(*"XVID")
    if realtime:
        frs,prop=execute()
        vw=cv2.VideoWriter("face_blur.avi",fourcc,
                           prop["fps"],prop["wh"],True)
        for fr in frs:
            vw.write(fr)
        vw.release()
        return
    a=os.listdir(".")
    for x in a[:]:
        if osp.isdir(x):
            a.remove(x)
        if osp.splitext(x)[-1]==".py":
            a.remove(x)
        if osp.splitext(x)[-1]==".dat":
            a.remove(x)
    for x in a:
        print(f"process {x}")
        frs,prop=execute(x)
        vw=cv2.VideoWriter(osp.splitext(x)[0]+"_blur"+osp.splitext(x)[1],prop["fcc"],
                           prop["fps"],prop["wh"],True)
        for fr in frs:
            vw.write(fr)
        vw.release()
if __name__=="__main__":
    master(False)