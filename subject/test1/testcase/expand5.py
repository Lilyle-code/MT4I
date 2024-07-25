import cv2
import numpy as np
#import matplotlib.pyplot as plt

# Gray scale
def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    #b = img[:, :, 1].copy()#变体33-CRP-46
    g = img[:, :, 1].copy()
    #g = img[:, :, 2].copy()#变体34-CRP-47
    r = img[:, :, 2].copy()

    # Gray scale
    #out = 0.2126 * r + 0.7152 * g + 0.0722 * b

    #out = 0.2126 / r + 0.7152 * g + 0.0722 * b#变体1-AOR-1

    #out = 0.2126 // r + 0.7152 * g + 0.0722 * b#变体2-AOR-2

    #out = 0.2126 ** r + 0.7152 * g + 0.0722 * b#变体3-AOR-3

    #out = 0.2126 * r - 0.7152 * g + 0.0722 * b#变体4-AOR-4
    out = 0.2126 * r + 0.7152 / g + 0.0722 * b#变体5-AOR-5
    #out = 0.2126 * r + 0.7152 // g + 0.0722 * b#变体6-AOR-6
    #out = 0.2126 * r + 0.7152 ** g + 0.0722 * b#变体7-AOR-7
    #out = 0.2126 * r + 0.7152 * g - 0.0722 * b#变体8-AOR-8
    #out = 0.2126 * r + 0.7152 * g + 0.0722 / b#变体9-AOR-9
    #out = 0.2126 * r + 0.7152 * g + 0.0722 // b#变体10-AOR-10
    #out = 0.2126 * r + 0.7152 * g + 0.0722 ** b#变体11-AOR-11
    #out = 1.2126 * r + 0.7152 * g + 0.0722 * b#变体35-CRP-49
    #out = 0.2126 * r + 1.7152 * g + 0.0722 * b#变体36-CRP-50
    #out = 0.2126 * r + 0.7152 * g + 1.0722 * b#变体37-CRP-51
    out = out.astype(np.uint8)

    return out

# Otsu Binalization
def otsu_binarization(img, th=128):
#def otsu_binarization(img, th=129):#变体38-CRP-52
    H, W = img.shape
    out = img.copy()

    max_sigma = 0
    #max_sigma = 1#变体39-CRP-53
    max_t = 0
    #max_t = 1#变体40-CRR-54

    # determine threshold
    for _t in range(1, 255):
    #for _t in range(2, 255):#变体41-CRP-55
    #for _t in range(1, 256):#变体42-CRP-56
        v0 = out[np.where(out < _t)]
        #v0 = out[np.where(out > _t)]#变体56-ROR-95
        #v0 = out[np.where(out <= _t)]#变体57-ROR-96
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        #m0 = np.mean(v0) if len(v0) > 1 else 0.#变体43-CRP-57
        #m0 = np.mean(v0) if len(v0) > 0 else 1.0#变体44-CRP-58
        #m0 = np.mean(v0) if len(v0) < 0 else 0.#变体58-ROR-97
        #m0 = np.mean(v0) if len(v0) >= 0 else 0.#变体59-CRP-98
        w0 = len(v0) / (H * W)
        #w0 = len(v0) // (H * W)#变体12-AOR-12
        #w0 = len(v0) * (H * W)#变体13-AOR-13
        #w0 = len(v0) / (H / W)#变体14-AOR-14
        #w0 = len(v0) / (H ** W)#变体15-AOR-16

        v1 = out[np.where(out >= _t)]
        #v1 = out[np.where(out <= _t)]#变体60-ROR-99
        #v1 = out[np.where(out > _t)]#变体61-ROR-100
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        #m1 = np.mean(v1) if len(v1) > 1 else 0.#变体45-CRP-59
        #m1 = np.mean(v1) if len(v1) > 0 else 1.0#变体46-CRP-60
        #m1 = np.mean(v1) if len(v1) < 0 else 0.#变体62-ROR-101
        #m1 = np.mean(v1) if len(v1) >= 0 else 0.#变体63-ROR-102
        w1 = len(v1) / (H * W)
        #w1 = len(v1) // (H * W)#变体16-AOR-17
        #w1 = len(v1) * (H * W)#变体17-AOR-18
        #w1 = len(v1) / (H / W)#变体18-AOR-19
        #w1 = len(v1) / (H ** W)#变体19-AOR-21
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        #sigma = w0 ** w1 * ((m0 - m1) ** 2)#变体20-AOR-24
        #sigma = w0 * w1 / ((m0 - m1) ** 2)#变体21-AOR-25
        #sigma = w0 * w1 // ((m0 - m1) ** 2)#变体22-AOR-26
        #sigma = w0 * w1 ** ((m0 - m1) ** 2)#变体23-AOR-27
        #sigma = w0 * w1 * ((m0 + m1) ** 2)#变体24-AOR-28
        #sigma = w0 * w1 * ((m0 - m1) * 2)#变体25-AOR-29
        #sigma = w0 * w1 * ((m0 - m1) ** 3)#变体47-CRP-61
        if sigma > max_sigma:
        #if not(sigma > max_sigma):#变体31-COI-42
        #if sigma < max_sigma:#变体64-COI-103
        #if sigma >= max_sigma:#变体65-COI-104
            max_sigma = sigma
            max_t = _t

    # Binarization
    print("threshold >>", max_t)
    th = max_t
    out[out < th] = 0
    #out[out < th] = 1#变体48-CRP-62
    #out[out > th] = 0#变体66-ROR-105
    #out[out <= th] = 0#变体67-ROR-106
    out[out >= th] = 255
    #out[out >= th] = 256#变体49-CRP-63
    #out[out <= th] = 255#变体68-ROR-107
    #out[out > th] = 255#变体69-ROR-108

    return out


# Morphology Dilate
def Morphology_Dilate(img, Dil_time=1):
    H, W = img.shape

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each dilate time
    out = img.copy()
    for i in range(Dil_time):
        tmp = np.pad(out, (1, 1), 'edge')
        #tmp = np.pad(out, (2, 1), 'edge')#变体50-CRP-74
        #tmp = np.pad(out, (1, 2), 'edge')#变体51-CRP-75
        for y in range(1, H):
        #for y in range(2, H):#变体52-CRP-78
            for x in range(1, W):
            #for x in range(2, W):#变体53-CRP-79
                if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                #if np.sum(MF / tmp[y - 1:y + 2, x - 1:x + 2]) >= 255:#变体26-AOR-35
                #if np.sum(MF // tmp[y - 1:y + 2, x - 1:x + 2]) >= 255:#变体27-AOR-36
                #if np.sum(MF ** tmp[y - 1:y + 2, x - 1:x + 2]) >= 255:#变体28-AOR-37
                #if np.sum(MF * tmp[y + 1:y + 2, x - 1:x + 2]) >= 255:#变体29-AOR-38
                #if np.sum(MF * tmp[y - 1:y + 2, x + 1:x + 2]) >= 255:#变体30-AOR-40
                #if not (np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) >= 255):#变体32-COI-45
                #if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) >= 256:#变体54-CRP-91
                #if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) <= 255:#变体70-ROR-112
                #if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) > 255:#变体71-ROR-113
                    out[y, x] = 255
                    #out[y, x] = 256#变体55-CRP-92

    return out


# Morphology Erode
def Morphology_Erode(img, Erode_time=1):
    H, W = img.shape
    out = img.copy()

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each erode
    for i in range(Erode_time):
        tmp = np.pad(out, (1, 1), 'edge')
        # erode
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                    out[y, x] = 0

    return out


def testAPI(str):
    # Read image
    img = cv2.imread(str).astype(np.float32)

    # Grayscale
    gray = BGR2GRAY(img)

    # Otsu's binarization
    otsu = otsu_binarization(gray)

    # Morphology - dilate
    #erode_result = Morphology_Erode(otsu, Erode_time=2)
    dilate_result = Morphology_Dilate(otsu,Dil_time=2)
    return dilate_result

#彩色->二值
def binary(img):
    gray = BGR2GRAY(img)
    otsu = otsu_binarization(gray)
    return otsu

#腐蚀算法
def erode(otsu):
    erode_result = Morphology_Erode(otsu, Erode_time=2)
    return erode_result
# Save result
'''cv2.imwrite("Black_and_white.jpg",otsu)
cv2.imshow("Black_and_white",otsu)
cv2.imwrite("erode_result.jpg", erode_result)
cv2.imshow("erode_result", erode_result)
cv2.imwrite("dilate_result.jpg", dilate_result)
cv2.imshow("dilate_result",dilate_result)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
