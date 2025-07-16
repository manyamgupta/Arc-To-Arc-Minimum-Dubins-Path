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
    pathType = LRL

    plotformat = SimpleNamespace(color='blue', linewidth=1, linestyle='-', marker='x')
    iniConf = np.array([0,0,0])
    rho = 1
    targRad = 2.2
    targRotOrient = +1
    
    ndisc = 1000
    iniConf = [0, 0, 0]
    arc2 = [2.5, 1.5, 0, 2*np.pi]
    
    thetaVec2 = np.linspace(arc2[2],arc2[3],ndisc)
    lenVec = np.zeros([ndisc])
    
    finConfVec = np.array([arc2[0]+targRad*cos(thetaVec2), arc2[1]+targRad*sin(thetaVec2), thetaVec2-targRotOrient*pi/2])
    
    # du.PlotArc(arc2[0:2], targRad, arc2[2:4], SimpleNamespace(color='green', linewidth=2, linestyle='-'))    


    for k in range(ndisc):            
        finConf = finConfVec[:,k]
        pathDub = dubins.path(iniConf, finConf, rho, pathType)
        if pathDub == None:
            lenVec[k] = 10000.
        else:
            lenVec[k] = pathDub.path_length()
            # du.PlotDubinsPath(pathDub, plotformat)
    
    minInd = np.argmin(lenVec)
    theta2_min = thetaVec2[minInd]
    
    finConf_min = np.array([arc2[0]+targRad*cos(theta2_min), arc2[1]+targRad*sin(theta2_min), theta2_min-targRotOrient*pi/2])
    pathDub_min = dubins.path(iniConf, finConf_min, rho, pathType)
    print("extremum is at alpha: ", theta2_min)
    infInds = np.argwhere(lenVec == 10000.)    
    for i in range(np.size(infInds,0)):
        lenVec[infInds[i]] = np.nan
    plt.figure()
    plt.plot(thetaVec2, lenVec)    

    plt.figure()
    utils.PlotCircle(arc2[0:2], targRad,SimpleNamespace(color='green', linewidth=1, linestyle='--'))
    du.PlotDubinsPath(pathDub_min, plotformat)
    # print("second segment length: ", pathDub_min.segment_length(1))
    plt.axis('equal')
    
    plt.show()
