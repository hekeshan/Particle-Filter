# -*- coding: utf-8 -*-
import numpy as np 
from matplotlib import pyplot as plt 
import cv2 as cv
import math 
import os.path

#产生一个 4 x Npop_particles 的矩阵，前两行为各元素值为 1：Npix_resolution(2)之间的产生的均匀分布的随机整数
def creat_particles(Npix_resolution,Npop_particles):
    X1 = np.random.randint(1,Npix_resolution[1]+1,size=(1,Npop_particles))
    X2 = np.random.randint(1,Npix_resolution[0]+1,size=np.shape(X1))
    X3 = np.zeros(shape=(2,Npop_particles))
    return np.vstack((X1,X2,X3))
#X[0:2, :] 各粒子在画面中的位置,X[2:4]各粒子的速度
def update_particles(F_update,Xstd_pos,Xstd_vec,X):
    N = np.shape(X)[1]
    X = np.dot(F_update,X)
    X[0:2,:] += Xstd_pos*np.random.randn(2,N)
    X[2:4,:] += Xstd_vec*np.random.randn(2,N)
    return X 

def calc_log_likelihood(Xstd_rgd,Xrgb_trgd,X,Y):
    Npix_h,Npix_w = np.shape(Y)[0],np.shape(Y)[1]
    N = np.shape(X)[1]
    L = np.zeros((1,N))
    Y = np.transpose(Y,(2,0,1))

    A = -math.log(math.sqrt(2*math.pi)*Xstd_rgd)
    B = -0.5/Xstd_rgd**2
    X = X.astype(np.int)

    for k in range(N):
        m,n = X[0,k],X[1,k]

        I,J = ((m>=0)&(m<Npix_h)),((n>=0) & (n<Npix_w))
        if I and J:
            C = Y[:,m,n]
            D = C-Xrgb_trgd
            D2 = np.dot(D,D.T)   #欧式距离
            L[0,k] = A+B*D2      #高斯似然，D2越小，L越大，注意B为负数
        else:
            L[0,k] = -np.inf
    return L

def resample_particle(X,L_log):
    #calculating Cumulative Distribution
    L = np.exp(L_log-np.max(L_log))
    Q = L/np.sum(L,1)
    R = np.cumsum(Q,1).flatten()
    #Generating Random Numbers
    N = np.shape(X)[1]
    #Resampling
    I = np.zeros((1,N),dtype=np.int64).flatten()
    for i in range(N):
        I[i] = np.where(np.random.rand()<=R)[0][0] 
    return X[:,I]

F_update = np.array([[1,0,1,0],
                    [0,1,0,1],
                    [0,0,1,0],
                    [0,0,0,1]])

Npop_particles = 4000
Xstd_rgd,Xstd_pos,Xstd_vec = 50,25,5
Xrgb_trgd = np.array([[0,0,255]])
#Loading Movie
cap = cv.VideoCapture(os.path.dirname(__file__)+"/Person.wmv")
Npix_resolution = [cap.get(cv.CAP_PROP_FRAME_WIDTH),cap.get(cv.CAP_PROP_FRAME_HEIGHT)]
fps = cap.get(cv.CAP_PROP_FPS)
#Video save
# Video = cv.VideoWriter("C:\\Users\\Administrator\\Desktop\\partical filter\\PF_Video_Python\\VideoTest.wmv", 
#                         cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, (640,480))
i = 0
#Object Tracking by Particle Filter
X = creat_particles(Npix_resolution,Npop_particles)   # 粒子初始化，在画面中产生均匀分布的随机粒子
while cap.isOpened() and i<cap.get(cv.CAP_PROP_FRAME_COUNT):#视频的最后一帧读取为None,舍弃
    #Getting Image
    ret,Y_k = cap.read()
    i +=1
    #Forecasting
    #通过状态模型预测  这里采用的是在上一时刻基础上叠加噪声
    X = update_particles(F_update,Xstd_pos,Xstd_vec,X)
    #Calculating log likelihood
    L = calc_log_likelihood(Xstd_rgd,Xrgb_trgd,X[0:2,:],Y_k)
    #Resampling
    X = resample_particle(X,L)
    #showing Image
    for (x,y) in zip(X[1,:],X[0,:]):
        cv.circle(Y_k,(int(x),int(y)),1,(255,0,0),2)
    cv.imshow("+++Showing Particles +++",Y_k)
    # Video.write(Y_k)    #Video Save
    cv.waitKey(1000//int(fps))

cap.release()
cv.destroyAllWindows()