# 1) Считать картинку с экрана
# 2) По событию выделить эталон
# 3) Получать координаты эталона
import numpy as np
import cv2 as cv
from numba import njit
import matplotlib.pyplot as plt 

ri = np.zeros((10,10),dtype=np.uint8)
use_kor = True

def autoCorrel(ri):
    tmp = np.zeros_like(ri)
    tmp1 = np.hstack((tmp,tmp ,tmp))
    tmp2 = np.hstack((tmp,ri,tmp))
    f = np.vstack((tmp1, tmp2, tmp1)).astype(float)
    for i in range(1,5):
        k = kor(f,ri, 1, i)
        minval, maxval, min_i, max_i = cv.minMaxLoc(k)
        k = (k-minval)/(maxval-minval)
        print(i, "Автокорреляция", minval, maxval)

        cv.imshow("{} Canvas".format(i), f.astype(np.uint8))
        cv.imshow("{} Samokorrel".format(i), k)

@njit(fastmath=True)
def raznostnaya(ci, ri, ci_k=3, ri_k=2):
        (M, N) = ci.shape
        (i0, j0) = ri.shape

        k1 = np.zeros((M-i0, N-j0), dtype=np.float64)

        for di in range(0, M-i0, ci_k):
            for dj in range(0, N-j0,ci_k):
                for i in range(0, i0, ri_k):
                    for j in range(0, j0, ri_k):
                        k1[di, dj] += np.abs(ri[i, j] - ci[i+di, j+dj])
        k1 /= i0*j0/ri_k/ri_k
        
        return k1

@njit(fastmath=True)
def kor(ci, ri, ci_k=3, ri_k=2):
        (M, N) = ci.shape
        (i0, j0) = ri.shape
        ci_m = ci.mean()
        ri_m = ri.mean()
        k1 = np.zeros((M-i0, N-j0), dtype=np.float64)

        for di in range(0, M-i0, ci_k):
            for dj in range(0, N-j0,ci_k):
                for i in range(0, i0, ri_k):
                    for j in range(0, j0, ri_k):
                        k1[di, dj] += (ri[i, j]-ri_m)*(ci[i+di, j+dj]-ci_m)
        k1 /= (i0*j0/ri_k/ri_k*np.sqrt(ri.var()*ci.var()))
        
        return k1


vid = cv.VideoCapture(0) 
vid.set(cv.CAP_PROP_FRAME_WIDTH, 320)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

first_tick = True

def main():
    global first_tick
    global ri
    
    while(vid.isOpened()):
        ret, frame = vid.read()
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if first_tick:
            (height, width , chanels) = frame.shape
            print(width, height)
            ri_width = 50
            ri_height = 50
            ri = frame_grey[int((height-ri_height)/2):int((height+ri_height)/2), int((width-ri_width)/2):int((width+ri_width)/2)]
            first_tick = False

        frame = cv.rectangle(frame,
                                (int((width+ri_width)/2), int((height+ri_height)/2)),
                                (int((width-ri_width)/2), int((height-ri_height)/2)),
                                (100,100,0),
                                2)
        
        frame_grey_ = frame_grey.astype(float)
        if (use_kor):
            k = kor(frame_grey_, ri)
            minval, maxval, min_i, max_i = cv.minMaxLoc(k)
            k = (k-minval)/(maxval-minval)

            frame = cv.rectangle(frame,
                                    max_i,
                                    (max_i[0]+ri_height, max_i[1]+ri_width),
                                    (0,50,127),
                                    2)
            frame = cv.arrowedLine(frame,
                                (max_i[0]+ri_height, max_i[1]+ri_width),
                                (int((width-ri_width)/2), max_i[1]+ri_width),
                                (50,50,50),
                                2)
            
            frame = cv.putText(frame, "max(K) = {:.2f}".format(maxval), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (180,10,10), 3, 2)
            cv.imshow("Raznostnaya", np.uint8(k*255.0))

        else:
            raz = kor(frame_grey_, ri)
            raz[raz == raz.min()] = raz.max()
            # raz = cv.matchTemplate(frame_grey, ri, cv.TM_SQDIFF)

            minval, maxval, min_i, max_i = cv.minMaxLoc(raz)
            if minval > 2:
                frame = cv.putText(frame, "! OBJECT LOST !".format(minval), (20, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (50,50,250), 3, 2)     
            raz = (raz-minval)/(maxval-minval)


            frame = cv.rectangle(frame,
                                    min_i,
                                    (min_i[0]+ri_height, min_i[1]+ri_width),
                                    (0,50,127),
                                    2)
            frame = cv.arrowedLine(frame,
                                (min_i[0]+ri_height, min_i[1]+ri_width),
                                (int((width-ri_width)/2), min_i[1]+ri_width),
                                (50,50,50),
                                2)
            
            frame = cv.putText(frame, "min(K) = {:.2f}".format(minval), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (180,10,10), 3, 2)
            cv.imshow("Raznostnaya", np.uint8(raz*255.0))
        cv.imshow('Video capture', frame)
        cv.imshow("Kernel",ri)
        
        
        res = cv.waitKey(1)
        if res & 0xFF == ord('q') or res == 27: 
            break
        if res & 0xFF == ord('s'): 
            ri = frame_grey[int((height-ri_height)/2):int((height+ri_height)/2), int((width-ri_width)/2):int((width+ri_width)/2)].astype(float)
            autoCorrel(ri)
            print("Обновлено")
    
    vid.release()
    cv.destroyAllWindows() 


if __name__ == "__main__":
    main()