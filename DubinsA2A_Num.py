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
from dataclasses import dataclass

@dataclass
class CandidatePath:
    pathType: str 
    angPos_arc1: float # angular position at first arc
    angPos_arc2: float # angular position at second arc
    segLengths: tuple
    
def PathRS(arc1, arc2, al1, rho):
    # Assumption: cenetr of the first arc (0,0)
    c_x = arc2.cntr_x
    c_y = arc2.cntr_y
    r1 = arc1.arc_radius
    r2 = arc2.arc_radius
    lx = c_x-(r1-rho)*np.cos(al1)
    ly = c_y-(r1-rho)*np.sin(al1)
    
    
    dist_oc2 = np.sqrt(lx**2 + ly**2)
    if np.abs((r2-rho)/(dist_oc2)) > 1:
        return np.nan, [np.nan, np.nan, np.nan, np.nan, np.nan ], []
    psi1 = np.arctan2(ly, lx)
    psi2 = np.arcsin((r2-rho)/dist_oc2)
    
    phi1 = np.mod(al1-psi1-psi2-np.pi/2, 2*pi)
    
    Ls = np.sqrt(lx**2 + ly**2 - (r2-rho)**2 )
    lengthLS = Ls + rho*phi1
    
    # inf_x = (r1-rho)*np.cos(al1) + rho*cos(psi1+psi2-np.pi/2)
    # inf_y = (r1-rho)*np.sin(al1) + rho*sin(psi1+psi2-np.pi/2)
    
    return lengthLS, [rho*phi1, Ls]

def A2AMinDubinsNum(arc1, arc2, rho, nd=1000):
    
    alVec1 = utils.AngularLinSpace(arc1.angPos_lb, arc1.angPos_ub, nd)
    alVec2 = utils.AngularLinSpace(arc2.angPos_lb, arc2.angPos_ub, nd)
    lenVec = np.zeros([nd, nd])
    iniConfVec = np.array([arc1.cntr_x + arc1.arc_radius*np.cos(alVec1), arc1.cntr_y+arc1.arc_radius*np.sin(alVec1), alVec1-pi/2])
    finConfVec = np.array([arc2.cntr_x + arc2.arc_radius*np.cos(alVec2), arc2.cntr_y+arc2.arc_radius*np.sin(alVec2), alVec2-pi/2])
    
    for j in range(nd):
        for k in range(nd):
            iniConf = iniConfVec[:,j]      
            finConf = finConfVec[:,k]
            pathDub = dubins.shortest_path(iniConf, finConf, rho)
            if pathDub:
                lenVec[j,k] = pathDub.path_length()                
            else:
                lenVec[j,k] = 1000000.
    minInd = np.argmin(lenVec)
    minPathLength = np.min(lenVec)
    inds2D = np.unravel_index(minInd, (nd,nd))
    alpha1_min = alVec1[inds2D[0]]
    alpha2_min = alVec2[inds2D[1]]      
    
    iniConf_min = np.array([arc1.cntr_x + arc1.arc_radius*np.cos(alpha1_min), arc1.cntr_y+arc1.arc_radius*np.sin(alpha1_min), alpha1_min-pi/2])
    finConf_min = np.array([arc2.cntr_x + arc2.arc_radius*np.cos(alpha2_min), arc2.cntr_y+arc2.arc_radius*np.sin(alpha2_min), alpha2_min-pi/2])
    
    pathDub_min = dubins.shortest_path(iniConf_min, finConf_min, rho) 
    segLengths = (pathDub_min.segment_length(0), pathDub_min.segment_length(1), pathDub_min.segment_length(2) )
    
    sp = CandidatePath(du.DubPathTypeNum2Str(pathDub_min.path_type()), alpha1_min,alpha2_min, segLengths)
    return minPathLength, sp

if __name__ == "__main__":

    LSL =0; LSR = 1; RSL = 2; RSR = 3; RLR = 4; LRL = 5; 
    pathType = RLR

    pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
    arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
    arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')

    # iniConf = np.array([0,0,0])
    # rho = 1
    # targRad = 2.5
    
    
    # ndisc = 100
    # arc1 = [0,0, .0, 2.]
    # arc2 = [9,3, 2., 4.]
    # thetaVec1 = np.linspace(arc1[2],arc1[3],ndisc)
    # thetaVec2 = np.linspace(arc2[2],arc2[3],ndisc)
    # lenVec = np.zeros([ndisc, ndisc])
    # iniConfVec = np.array([arc1[0]+targRad*cos(thetaVec1), arc1[1]+targRad*sin(thetaVec1), thetaVec1+pi/2])
    # finConfVec = np.array([arc2[0]+targRad*cos(thetaVec2), arc2[1]+targRad*sin(thetaVec2), thetaVec2+pi/2])
    
    # du.PlotArc(arc1[0:2], targRad, arc1[2:4], SimpleNamespace(color='green', linewidth=2, linestyle='-'))
    # du.PlotArc(arc2[0:2], targRad, arc2[2:4], SimpleNamespace(color='green', linewidth=2, linestyle='-'))
    # utils.PlotCircle(arc1[0:2], targRad,SimpleNamespace(color='green', linewidth=1, linestyle='--'))
    # utils.PlotCircle(arc2[0:2], targRad,SimpleNamespace(color='green', linewidth=1, linestyle='--'))

    # for j in range(ndisc):
    #     for k in range(ndisc):
    #         iniConf = iniConfVec[:,j]      
    #         finConf = finConfVec[:,k]
    #         pathDub = dubins.path(iniConf, finConf, rho, pathType)
    #         if pathDub == None:
    #             lenVec[j,k] = np.nan
    #         else:
    #             lenVec[j,k] = pathDub.path_length()
    #             # du.PlotDubinsPath(pathDub, plotformat)
    
    # minInd = np.argmin(lenVec)
    # inds2D = np.unravel_index(minInd, (ndisc,ndisc))
    # theta1_min = thetaVec1[inds2D[0]]
    # theta2_min = thetaVec2[inds2D[1]]
    # iniConf_min = np.array([arc1[0]+targRad*cos(theta1_min), arc1[1]+targRad*sin(theta1_min), theta1_min+pi/2])
    # finConf_min = np.array([arc2[0]+targRad*cos(theta2_min), arc2[1]+targRad*sin(theta2_min), theta2_min+pi/2])
    # pathDub_min = dubins.path(iniConf_min, finConf_min, rho, pathType)
    # du.PlotDubinsPath(pathDub_min, plotformat)
    # plt.axis('equal')
    
    # # X, Y = np.meshgrid(thetaVec1, thetaVec2)
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # Z = lenVec

    # # ax.plot_surface(X, Y, Z)
    # # ax.set_xlabel(r'$\theta_1$')
    # # ax.set_ylabel(r'$\theta_2$')
    # # ax.set_zlabel('path length')

    # plt.show()
    
    ######################### Check RS minimum ##############################
    
    arc1 = utils.Arc(0,0, 2.5, 5.01, 6.28)
    arc2 = utils.Arc(7., 2.5, 2.5, 0.01, 6.28)
    rho = 1
    nd = 1000
    r1 = arc1.arc_radius        
    r2 = arc2.arc_radius  
    psi3 = np.arctan2(arc2.cntr_y, arc2.cntr_x)
    pathNum = 1
    alVsLen = np.zeros([nd,4])
    alVec = utils.AngularLinSpace(arc1.angPos_lb, arc1.angPos_ub, nd)
    distVec = np.ones(nd)*np.nan
    lenPrVec = np.ones(nd)*np.nan
    lenPrVec2 = np.ones(nd)*np.nan

    lineSeg = np.array([[arc1.cntr_x, arc1.cntr_y], [arc2.cntr_x, arc2.cntr_y]])
    
    # alVec = [6]
    LengthVec = np.zeros(nd)
    for indx, al1 in enumerate(alVec):
        

        pathLen, segLengths = PathRS(arc1, arc2, al1, rho)
        
        if np.isfinite(pathLen):            
            LengthVec[indx] = pathLen
    
    minInd = np.argmin(LengthVec)
    minAl = alVec[minInd]
    # minAl = alVec[0]
    iniPos = np.array([r1*np.cos(minAl), r1*np.sin(minAl)])
    iniHdng = minAl-np.pi/2
    pathLen, segLengths = PathRS(arc1, arc2, minAl, rho)
    print('segLengths: ', segLengths)
    du.PlotDubPathSegments([iniPos[0], iniPos[1], iniHdng], 'RS', segLengths[0:2], rho, pathfmt)
    utils.PlotArc(arc1, arcfmt)
    utils.PlotArc(arc2, arcfmt)          
    plt.axis('equal')
    
    plt.figure()
    plt.plot(alVec, LengthVec)
    
    plt.show()

