import cv2
import numpy as np
import matplotlib.pyplot as plt


def sobel(src):
    sobel_x = np.array([[-1, -2, -1], 
                        [ 0,  0,  0], 
                        [ 1,  2,  1]]) 
    sobel_y = np.array([[1, 0, -1], 
                        [2, 0, -2], 
                        [1, 0, -1]])
    xdir = cv2.convertScaleAbs(cv2.filter2D(src, cv2.CV_32F, sobel_x))
    ydir = cv2.convertScaleAbs(cv2.filter2D(src, cv2.CV_32F, sobel_y))
    return cv2.addWeighted(xdir, 1, ydir, 1, 0);

def prewitt(src):
    prewitt_x = np.array([[-1, -1, -1], 
                          [ 0,  0,  0], 
                          [ 1,  1,  1]]) 
    prewitt_y = np.array([[1, 0, -1], 
                          [1, 0, -1], 
                          [1, 0, -1]]) 
    xdir = cv2.convertScaleAbs(cv2.filter2D(src, -1, prewitt_x))
    ydir = cv2.convertScaleAbs(cv2.filter2D(src, -1, prewitt_y))
    return cv2.addWeighted(xdir, 1, ydir, 1, 0);
    
def roberts(src):
    roberts_x = np.array([[-1, 0, 0], 
                          [ 0, 1, 0], 
                          [ 0, 0, 0]]) 
    roberts_y = np.array([[0, 0, -1], 
                          [0, 1,  0], 
                          [0, 0,  0]]) 
    xdir = cv2.convertScaleAbs(cv2.filter2D(src, -1, roberts_x))
    ydir = cv2.convertScaleAbs(cv2.filter2D(src, -1, roberts_y))
    return cv2.addWeighted(xdir, 1, ydir, 1, 0);
    
def scharr(src):
    scharr_x = np.array([[ 3, 10, 3], 
                         [ 0,  0, 0], 
                         [-3,-10,-3]]) 
    scharr_y = np.array([[ 3, 0,  -3], 
                         [10, 0, -10], 
                         [ 3, 0,  -3]])
    xdir = cv2.convertScaleAbs(cv2.filter2D(src, -1, scharr_x))
    ydir = cv2.convertScaleAbs(cv2.filter2D(src, -1, scharr_y))
    return cv2.addWeighted(xdir, 1, ydir, 1, 0);

def canny(src):
    return cv2.Canny(src, 25, 50)
    
def blob(src):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.thresholdStep = 0.01
    params.maxThreshold = 1
    params.minArea = 0.01;
    params.maxArea = 100;
    params.filterByArea = True;
    params.filterByColor = True
    params.blobColor = 0

    detector = cv2.SimpleBlobDetector_create(params)
    kp = detector.detect(src)

    return cv2.drawKeypoints(src, kp, np.zeros((1,1)), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def laplacian(src):
    return cv2.Laplacian(src, -1)
def LoG(src):
    mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    gaussian = cv2.GaussianBlur(src, (5, 5), 0) 
    return cv2.filter2D(gaussian, -1, mask)
def DoG(src):
    height, width = src.shape
    gaussian1 = cv2.GaussianBlur(src, (5, 5), 1.6) 
    gaussian2 = cv2.GaussianBlur(src, (5, 5), 1) 
    DoG = np.zeros_like(src) 
    for i in range(height): 
        for j in range(width): 
            DoG[i][j] = float(gaussian1[i][j]) - float(gaussian2[i][j])

    return DoG


if __name__ == "__main__": 
    img1_color = cv2.imread('dataset/img1.jpg')
    img2_color = cv2.imread('dataset/img2.jpg')
    img3_color = cv2.imread('dataset/img3.jpg')
    img4_color = cv2.imread('dataset/img4.jpg')

    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    img3_gray = cv2.cvtColor(img3_color, cv2.COLOR_BGR2GRAY)
    img4_gray = cv2.cvtColor(img4_color, cv2.COLOR_BGR2GRAY)

    img_color = img4_color
    img_gray = img4_gray
    
    fig, ax = plt.subplots(3,3, figsize=(10,10), sharey=True)
    fig.canvas.manager.set_window_title('compare edge detector')
    
    ax[0][0].axis('off')
    ax[0][1].axis('off')
    ax[0][2].axis('off')
    ax[1][0].axis('off')
    ax[1][1].axis('off')
    ax[1][2].axis('off')
    ax[2][0].axis('off')
    ax[2][1].axis('off')
    ax[2][2].axis('off')

    ax[0][0].set_title('original')
    ax[0][1].set_title('sobel')
    ax[0][2].set_title('prewitt')
    ax[1][0].set_title('roberts')
    ax[1][1].set_title('scharr')
    ax[1][2].set_title('canny')
    ax[2][0].set_title('blob')
    ax[2][1].set_title('LoG')
    ax[2][2].set_title('DoG')
    
    ax[0][0].imshow(img_color, aspect="auto")
    ax[0][1].imshow(sobel(img_gray), aspect="auto")
    ax[0][2].imshow(prewitt(img_gray), aspect="auto")
    ax[1][0].imshow(roberts(img_gray), aspect="auto")
    ax[1][1].imshow(scharr(img_gray), aspect="auto")
    ax[1][2].imshow(canny(img_gray), aspect="auto")
    ax[2][0].imshow(blob(img_gray), aspect="auto")
    ax[2][1].imshow(LoG(img_gray), aspect="auto")
    ax[2][2].imshow(DoG(img_gray), aspect="auto")
    
    # dstName = "data/myFig.png"
    # plt.savefig(dstName, bbox_inches="tight")
    fig.tight_layout()
    plt.show()

