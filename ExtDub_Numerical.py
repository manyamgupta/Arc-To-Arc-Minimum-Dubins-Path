# Shortest Dubins LSR path to a line segment
# Author: Satya Gupta Manyam
import numpy as np
from numpy import pi,cos,sin
import matplotlib.pyplot as plt
import dubins 
import dubutils as du
import utils
from collections import namedtuple
from types import SimpleNamespace

if __name__ == "__main__":

    LSL =0; LSR = 1; RSL = 2; RSR = 3; RLR = 4; LRL = 5; 
    pathType = RLR

    plotformat = SimpleNamespace(color='blue', linewidth=1, linestyle='-', marker='x')
    iniConf = np.array([0,0,0])
    rho = 1
    r1 = 2.5
    r2 = 2.8
    
    
    ndisc = 1000
    c_x = 0.7
    # arc1 = [0,0, .1, 3.5]
    # arc2 = [c_x, 0, 4.1, 7.2]
    # A2ADub = Arc2ArcDubins([0, 0], 2.5, [0.01, 6.28], [4.5, -4], 2.8, [0.01, 6.28], 1) 
    
    arc1 = [0,0, .01, 6.2]
    arc2 = [-4.5, 4, .1, 6.2]

    thetaVec1 = np.linspace(arc1[2],arc1[3],ndisc)
    thetaVec2 = np.linspace(arc2[2],arc2[3],ndisc)
    lenVec = np.zeros([ndisc, ndisc])
    iniConfVec = np.array([arc1[0]+r1*cos(thetaVec1), arc1[1]+r1*sin(thetaVec1), thetaVec1-pi/2])
    finConfVec = np.array([arc2[0]+r2*cos(thetaVec2), arc2[1]+r2*sin(thetaVec2), thetaVec2-pi/2])
    
    for j in range(ndisc):
        for k in range(ndisc):
            iniConf = iniConfVec[:,j]      
            finConf = finConfVec[:,k]
            pathDub = dubins.path(iniConf, finConf, rho, pathType)
            if pathDub == None:
                lenVec[j,k] = -10000.
            else:
                lenVec[j,k] = pathDub.path_length()
                # du.PlotDubinsPath(pathDub, plotformat)
    
    minInd = np.argmax(lenVec)
    inds2D = np.unravel_index(minInd, (ndisc,ndisc))
    theta1_min = thetaVec1[inds2D[0]]
    theta2_min = thetaVec2[inds2D[1]]
    print("theta1_min: ", theta1_min)
    print("theta2_min: ", theta2_min)
    iniConf_min = np.array([arc1[0]+r1*cos(theta1_min), arc1[1]+r1*sin(theta1_min), theta1_min-pi/2])
    finConf_min = np.array([arc2[0]+r2*cos(theta2_min), arc2[1]+r2*sin(theta2_min), theta2_min-pi/2])
    
    pathDub_min = dubins.path(iniConf_min, finConf_min, rho, pathType) 
    minPathLength = pathDub_min.path_length()
    seg1_len = pathDub_min.segment_length(0)
    seg2_len = pathDub_min.segment_length(1)
    seg3_len = pathDub_min.segment_length(2)
    segLengths = [seg1_len, seg2_len, seg3_len]
    print("segLengths: ", segLengths)
    
    ##### chck RLR min ######
    
    beta = 2*np.pi-seg2_len/rho
    lamda = rho*np.sin(beta/2)
    OQ1 = np.sqrt((r1-rho)**2-rho**2+lamda**2)
    CQ2 = np.sqrt((r2-rho)**2-rho**2+lamda**2)
    d = c_x
    exp1 = OQ1+CQ2+d-4*rho*lamda
    print("exp1: ", exp1)
    
    zeta = (r1-rho)**2+(r2-rho)**2
    exp2 = (4*rho*lamda-d)**4+ zeta**2+(4*rho*lamda-d)**2*(-2*zeta-4*lamda**2+4*rho**2)-4*(r1-rho)**2*(r2-rho)**2
    print("exp2: ", exp2)
    
    c1 = 256*rho**4-64*rho**2
    c2 = -256*rho**3*d+32*rho*d
    c3 = 96*rho**2*d**2-32*rho**2*zeta+64*rho**4-4*d**2
    c4 = -16*rho*d**3+16*rho*d*zeta-32*rho**3*d
    c5 = d**4 + zeta**2 - 2*zeta*d**2 + 4*rho**2*d**2 -4*(r1-rho)**2*(r2-rho)**2
    
    exp3 = c1*lamda**4 + c2*lamda**3 + c3*lamda**2 + c4*lamda + c5
    print("exp3: ", exp3)
    
    
    config_inf1 = pathDub_min.sample(seg1_len)
    config_inf2 = pathDub_min.sample(seg1_len+seg2_len)
    
    print("minPathLength: ", minPathLength)
    
    du.PlotArc(arc1[0:2], r1, arc1[2:4], SimpleNamespace(color='green', linewidth=2, linestyle='-'))
    du.PlotArc(arc2[0:2], r2, arc2[2:4], SimpleNamespace(color='green', linewidth=2, linestyle='-'))
    utils.PlotCircle(arc1[0:2], r1,SimpleNamespace(color='green', linewidth=1, linestyle='--'))
    utils.PlotCircle(arc2[0:2], r2,SimpleNamespace(color='green', linewidth=1, linestyle='--'))

    du.PlotDubinsPath(pathDub_min, plotformat)
    plt.scatter(config_inf1[0], config_inf1[1],marker='x')
    plt.scatter(config_inf2[0], config_inf2[1],marker='x')
    plt.plot([arc1[0], arc2[0]],[arc1[1], arc2[1]])
    plt.axis('equal')
    
    X, Y = np.meshgrid(thetaVec1, thetaVec2)
    X = np.transpose(X)
    Y = np.transpose(Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Z = lenVec

    infInds = np.argwhere(Z == -10000.)
    
    for i in range(np.size(infInds,0)):
        Z[infInds[i][0], infInds[i][1]] = np.nan
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel('path length')

    plt.show()

