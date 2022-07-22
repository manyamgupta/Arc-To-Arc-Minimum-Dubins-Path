# Shortest Dubins path Arc to Arc
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
    targRad = 2.5
    
    
    ndisc = 100
    arc1 = [0,0, .0, 2.]
    arc2 = [9,3, 2., 4.]
    thetaVec1 = np.linspace(arc1[2],arc1[3],ndisc)
    thetaVec2 = np.linspace(arc2[2],arc2[3],ndisc)
    lenVec = np.zeros([ndisc, ndisc])
    iniConfVec = np.array([arc1[0]+targRad*cos(thetaVec1), arc1[1]+targRad*sin(thetaVec1), thetaVec1+pi/2])
    finConfVec = np.array([arc2[0]+targRad*cos(thetaVec2), arc2[1]+targRad*sin(thetaVec2), thetaVec2+pi/2])
    
    du.PlotArc(arc1[0:2], targRad, arc1[2:4], SimpleNamespace(color='green', linewidth=2, linestyle='-'))
    du.PlotArc(arc2[0:2], targRad, arc2[2:4], SimpleNamespace(color='green', linewidth=2, linestyle='-'))
    utils.PlotCircle(arc1[0:2], targRad,SimpleNamespace(color='green', linewidth=1, linestyle='--'))
    utils.PlotCircle(arc2[0:2], targRad,SimpleNamespace(color='green', linewidth=1, linestyle='--'))

    for j in range(ndisc):
        for k in range(ndisc):
            iniConf = iniConfVec[:,j]      
            finConf = finConfVec[:,k]
            pathDub = dubins.path(iniConf, finConf, rho, pathType)
            if pathDub == None:
                lenVec[j,k] = np.nan
            else:
                lenVec[j,k] = pathDub.path_length()
                # du.PlotDubinsPath(pathDub, plotformat)
    
    minInd = np.argmin(lenVec)
    inds2D = np.unravel_index(minInd, (ndisc,ndisc))
    theta1_min = thetaVec1[inds2D[0]]
    theta2_min = thetaVec2[inds2D[1]]
    iniConf_min = np.array([arc1[0]+targRad*cos(theta1_min), arc1[1]+targRad*sin(theta1_min), theta1_min+pi/2])
    finConf_min = np.array([arc2[0]+targRad*cos(theta2_min), arc2[1]+targRad*sin(theta2_min), theta2_min+pi/2])
    pathDub_min = dubins.path(iniConf_min, finConf_min, rho, pathType)
    du.PlotDubinsPath(pathDub_min, plotformat)
    plt.axis('equal')
    
    # X, Y = np.meshgrid(thetaVec1, thetaVec2)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # Z = lenVec

    # ax.plot_surface(X, Y, Z)
    # ax.set_xlabel(r'$\theta_1$')
    # ax.set_ylabel(r'$\theta_2$')
    # ax.set_zlabel('path length')

    plt.show()

