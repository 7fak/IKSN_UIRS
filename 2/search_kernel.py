# 1) Считать картинку с экрана
# 2) По событию выделить эталон
# 3) Получать координаты эталона
import numpy as np
import cv2 as cv
from numba import njit
import matplotlib.pyplot as plt 

ri = np.zeros((10,10),dtype=np.uint8)

def geometry_autoCorrel(mass1, corr_0):
    # corr_0 - пороговое значение корреляции для расчета # радиуса корреляции
    res = []
    res.append([])
    res.append(corr_0)
    res.append(1)
    res.append(1)

    base_i = mass1.shape[0]
    base_j = mass1.shape[1]
    virtual_surface = np.zeros((base_i*3,base_j*3))
    virtual_surface[...] = 255
    virtual_surface[base_i:2*base_i,base_j:2*base_j] = mass1
    #plt.imshow(virtual_surface, origin="upper", cmap='gray', vmin = 0)
    #plt.show()
    res[0] = CorrelMatrix(mass1, virtual_surface)#,1,1,corr_0,0.6) # корреляционная матрица

    kor_func

    # срезы автокорреляции
    j_max = int(res[0].shape[1]/2)
    i_max = int(res[0].shape[0]/2)
    # срез вдоль i
    i_slice=np.round(res[0][:,j_max], 1)
    #plt.plot(i_slice)
    #plt.show()
    i_slice=i_slice.tolist()
    c1 = 0 # левая граница радиуса корреляции
    c1 = i_slice.index(corr_0)
    # срез вдоль j
    #j_slice=list(res[0][i_max,:])
    #c2 = 0 # правя граница радиуса корреляции
    #c2 = i_slice.index(corr_0)
    res[2] = np.asarray(i_slice).shape[0]-2*c1 # радиус корреляции по i
    res[3] = res[2] # радиус корреляции по j
    return res 

@njit(parallel=True)
def kor(ci, ri, ci_k=6, ri_k=3, use_korrel=True, use_diff=True, last_index = (0,0)):
        (M, N) = ci.shape
        (i0, j0) = ri.shape


        k = np.zeros((M-i0, N-j0), dtype=np.float64)
        k1 = np.zeros((M-i0, N-j0), dtype=np.float64)+1e4

        m_ci = ci.mean()
        m_ri = ri.mean()


        for di in range(0, M-i0, ci_k):
            for dj in range(0, N-j0,ci_k):
                k1[di, dj] = 0
                for i in range(0, i0, ri_k):
                    for j in range(0, j0, ri_k):
                        k[di, dj] += (ri[i,j] - m_ri)*(ci[di + i, dj + j] - m_ci)
                        k1[di, dj] += np.abs(ri[i, j] - ci[i+di, j+dj])
        k1 /= ri.size/(ri_k**2)
        k /= ri.size/(ri_k**2)
        
        return (k,k1)

vid = cv.VideoCapture('./2/try2.mp4') 
#vid = cv.VideoCapture(0) 
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
        
        #frame = frame.astype(float)
        #ri = ri.astype(float)
        
        #korrel,raz = kor(frame_grey, ri)
        korrel = cv.matchTemplate(frame_grey, ri, cv.TM_CCORR_NORMED)
        raz = cv.matchTemplate(frame_grey, ri, cv.TM_SQDIFF)

        frame = frame.astype(np.uint8)

        minval, maxval, min_i, max_i = cv.minMaxLoc(korrel)
        korrel = (korrel-minval)/(maxval-minval)
        
        frame = cv.rectangle(frame,
                                max_i,
                                (max_i[0]+ri_height, max_i[1]+ri_width),
                                (255,0,0),
                                3)
        

        minval, maxval, min_i, max_i = cv.minMaxLoc(raz)
        raz = (raz-minval)/(maxval-minval)

        frame = cv.rectangle(frame,
                                min_i,
                                (min_i[0]+ri_height, min_i[1]+ri_width),
                                (0,255,0),
                                1)
        ri_ = frame_grey[ min_i[0]:min_i[0]+ri_height, min_i[1]:min_i[1]+ri_width]

        
        cv.imshow('Video capture', frame)
        cv.imshow("Kernel",ri)
        cv.imshow("Raznostnaya", np.uint8(raz*255.0))
        cv.imshow("Korrelatsionnaya", np.uint8(korrel/korrel.max()*255))
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 

        res = cv.waitKey(1)
        if res & 0xFF == ord('q'): 
            break
        if res & 0xFF == ord('s'): 
            ri = frame_grey[int((height-ri_height)/2):int((height+ri_height)/2), int((width-ri_width)/2):int((width+ri_width)/2)]
            print("Обновлено")
    
        # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv.destroyAllWindows() 

if __name__ == "__main__":
    main()