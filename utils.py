# Author: Satyanarayana Gupta Manyam
# from turtle import end_fill
from numpy import pi,cos,sin
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
# from collections import namedtuple

defaultFmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
from dataclasses import dataclass

    
@dataclass
class Arc:
    cntr_x: float
    cntr_y: float
    arc_radius: float
    angPos_lb: float # angular position lower bound
    angPos_ub: float # angular position upper bound

def DubPathReflection(pathMode):
    pathRef = ''
    for p in pathMode:
        if p == 'L':
            pathRef += 'R'
        elif p == 'R':
            pathRef += 'L'
        elif p == 'S':
            pathRef += 'S'
            
    return pathRef
def CheckFeasibility(arc1, arc2, al1, rho, segLengths, pathMode):
    
    if InInt(arc1.angPos_lb, arc1.angPos_ub, al1):
        iniConf = (arc1.cntr_x+arc1.arc_radius*np.cos(al1), arc1.cntr_y+arc1.arc_radius*np.sin(al1), al1-np.pi/2)
        
        for k in range(len(pathMode)):
            iniConf = MoveAlongSegment(iniConf, segLengths[k], pathMode[k], rho)
            # print('iniConf: ', iniConf)
        al2 = iniConf[2]+np.pi/2
        if np.abs(np.sqrt((iniConf[0]-arc2.cntr_x)**2+(iniConf[1]-arc2.cntr_y)**2)-arc2.arc_radius) <1e-4 and InInt(arc2.angPos_lb, arc2.angPos_ub, al2):
            return True

    return False


def MoveAlongSegment(startConf, segLength, segType, rho):
    
    t1 = startConf[2]
    if segType == 'S':
        pt2 = (startConf[0] + segLength*np.cos(t1), startConf[1] + segLength*np.sin(t1))
        finalConf = (pt2[0], pt2[1], t1)
    elif segType == 'L' or segType == 'R':

        rotSense = 1 if segType =='L' else -1
        center = (startConf[0] + rho*np.cos(t1+rotSense*pi/2), startConf[1] + rho*np.sin(t1+rotSense*pi/2))

        t2 = np.mod(t1  + rotSense*segLength/rho, 2*pi)
        al_final = t2 -rotSense*pi/2 

        tc_x = center[0]+rho*np.cos(al_final)
        tc_y = center[1]+rho*np.sin(al_final)
        
        
        finalConf = (tc_x, tc_y, t2)
    else:
        raise Exception('Error: Ineligible turn, each letter can be L or R or S. Let me let you go')

    
    return finalConf

def IntersectionLineSegments(p1,p2,p3,p4):
    # https://math.stackexchange.com/questions/3176543/intersection-point-of-2-lines-defined-by-2-points-each
    n = p2-p1
    m = p3-p4
    p = p3-p1

    D = n[0]*m[1]-n[1]*m[0]
    if np.abs(D) < 1e-6:
        return [np.nan, np.nan]
    Qx = m[1]*p[0]-m[0]*p[1]
    t = Qx/D
    
    intPt = p1+t*(p2-p1)

    return intPt

def RotateVec(vec, theta ):
    # rotates the vec in ccw direction for an angle of theta
    rotMat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(rotMat, np.array(vec))

def RotateVec2(vec, theta ):
    # rotates the vec in ccw direction for an angle of theta
    # returns tuple
    return (np.cos(theta)*vec[0]-np.sin(theta)*vec[1], np.sin(theta)*vec[0]+np.cos(theta)*vec[1])
    

def DistPtToLineSeg(pt, lineSeg):
    # perpendicular distance from point to linesegment
    lineSeg  =np.array(lineSeg)    
    A = lineSeg[0]; B = lineSeg[1]
    lenAB = np.linalg.norm(A-B)
    triangleArea = abs((B[0]-A[0])*(A[1]-pt[1]) - (B[1]-A[1])*(A[0]-pt[0]))
    
    return triangleArea/lenAB
    
def CheckPtLiesInsideLineSeg(a,b,c):
# check if point c lies inside line segment ab, assume abc are collinear

    if np.dot(c-a,b-c)>0:
        return True
    else:
        return False
    
    
def RotateCoords(coords, theta):
    R = np.matrix([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])  
    coords = np.matrix([coords[0], coords[1]])
    corods = coords.T
    coordsRot = R*corods
    
    return [coordsRot[0,0], coordsRot[1,0]]

# def PlotCircle(C, r,col):
#     alVec = np.linspace(0,2*pi,1000)

#     tc_x = C[0]+r*cos(alVec)
#     tc_y = C[1]+r*sin(alVec)
#     plt.plot(tc_x, tc_y, col,zorder=0) 

def AngularLinSpace(t_l, t_u, nd):
    
    t_l = np.mod(t_l, 2*np.pi)
    t_u = np.mod(t_u, 2*np.pi)
    
    if t_u ==0:
        return np.linspace(t_l, 2*np.pi, nd)
    elif InInt(t_l, t_u, 0) and t_l != 0:
        nd1 = int(nd*(Angdiff(t_l,2*np.pi-2*np.pi/nd,)/Angdiff(t_l, t_u)))
        nd2 = nd-nd1
        angspace1 = np.linspace(t_l,2*np.pi-2*np.pi/nd, nd1)
        angspace2 = np.linspace(0,t_u, nd2)
        return np.concatenate((angspace1, angspace2), axis=0)
    else:
        return np.linspace(t_l,t_u, nd)
    

def PlotArc(arc,fmt):

    theta_l = np.mod(arc.angPos_lb, 2*np.pi)
    theta_u = np.mod(arc.angPos_ub, 2*np.pi)
    r = arc.arc_radius
    
    alVec = AngularLinSpace(theta_l,theta_u,100)

    tc_x = arc.cntr_x+r*cos(alVec)
    tc_y = arc.cntr_y+r*sin(alVec)
    plt.scatter(arc.cntr_x, arc.cntr_y, marker='x', color=fmt.color)
    plt.plot(tc_x, tc_y, fmt.color, linewidth=fmt.linewidth, linestyle=fmt.linestyle) 
    return

def Angdiff(ti, tf ):
#    Angular difference from ti to tf, the interval is assumed to be ccw
    ti = np.mod(ti, 2*pi)
    tf = np.mod(tf, 2*pi)
    
    if ti == tf:
        return 0
    elif (InInt(ti, tf, 0)):
        diff = tf+(2*pi-ti)
    else:
        diff = tf-ti
        
    diff = np.abs(diff)        
    
    return diff
    
def InInt(lb, ub, t ):   
    # Checks if t is in the interval (lb, ub), interval goes ccw from lb to ub
    lb = np.mod(lb, 2*pi)
    ub = np.mod(ub, 2*pi)
    t = np.mod(t, 2*pi) 
    if lb==ub:
        print("improper interval, lb and ub cannot be the same")
        return False
    elif (lb > ub):
        if(t >= lb or t <= ub):
            return True
        else:
            return False
    else:
        if(t >= lb and t <= ub):
            return True
        else:
            return False
            
def MidAng(lb, ub ):  
    # finds the middle angle between lb and up, going ccw from lb to ub
    lb = np.mod(lb, 2*pi)
    ub = np.mod(ub, 2*pi)
    
    if lb == 0:
        case =1
    elif ub ==0:
        case =2
    elif InInt(lb, ub, 0 ):
        case =2
    else:
        case =1
    
    if case ==1:
        midang = (lb+ub)/2
        midang = np.mod(midang, 2*pi)
        return midang
    elif case==2:
        midang = (lb-2*pi+ub)/2
        midang = np.mod(midang, 2*pi)
        return midang     
               
    # if InInt(lb, ub, 0 ):
        
    #     midang = (lb-2*pi+ub)/2
    #     midang = np.mod(midang, 2*pi)
    #     return midang
        
    # else:
        
    #     midang = (lb+ub)/2
    #     midang = np.mod(midang, 2*pi)
    #     return midang
          	
def PlotPolygon(vertices, segments, fmt):

    vertices = np.array(vertices)
    segments = np.array(segments)
    numSegs = np.size(segments, 0)

    for i in range(numSegs):
        seg = segments[i,:]
        vertsSeg = vertices[seg,:]
        plt.plot(vertsSeg[:,0], vertsSeg[:,1], color=fmt.color, linewidth=fmt.linewidth)

    return

def PlotCircle(C, r,fmt):
    alVec = np.linspace(0,2*pi,1000)

    tc_x = C[0]+r*cos(alVec)
    tc_y = C[1]+r*sin(alVec)
    plt.plot(tc_x, tc_y, color=fmt.color, linewidth=fmt.linewidth, linestyle=fmt.linestyle,zorder=0) 
    plt.scatter(C[0], C[1],marker='x')

    return

# def PlotArc(C, r, phis,fmt):

#     alVec = np.linspace(phis[0],phis[1],100)

#     tc_x = C[0]+r*cos(alVec)
#     tc_y = C[1]+r*sin(alVec)
#     plt.scatter(C[0], C[1], color=fmt.color, marker=fmt.marker)
#     plt.plot(tc_x, tc_y, color=fmt.color, linewidth=fmt.linewidth, linestyle=fmt.linestyle) 

#     return

def PlotArrow(p1, hdng, arrLen, fmt):

    # p2 = p1 + arrLen*np.array([cos(hdng), sin(hdng)])
    dx = arrLen*np.cos(hdng)
    dy = arrLen*np.sin(hdng)

    # plt.plot([p1[0],p2[0]], [p1[1],p2[1]], color=fmt.color, linewidth=fmt.linewidth, linestyle=fmt.linestyle)     
    plt.arrow(p1[0],p1[1],dx,dy,head_width=.1*np.sqrt(dx**2+dy**2),color=fmt.color,linewidth=fmt.linewidth, linestyle=fmt.linestyle)
    return

def PlotLineSeg(p1, p2, fmt=defaultFmt):

    plt.plot([p1[0],p2[0]], [p1[1],p2[1]], color=fmt.color, linewidth=fmt.linewidth, linestyle=fmt.linestyle) 

    return

def PlotParalellogram(prlGrm, fmt=defaultFmt):
    prlGrm.append(prlGrm[0])
    for k in range(4):
        PlotLineSeg(prlGrm[k], prlGrm[k+1], fmt)
    
    return

