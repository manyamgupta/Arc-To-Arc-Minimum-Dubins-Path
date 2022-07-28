import numpy as np
from numpy import pi,cos,sin
import matplotlib.pyplot as plt
import dubins
import dubutils as du 
import utils
from types import SimpleNamespace

def LocalMinLS(arc1, arc2, rho):
    # Local minimum of LS path from arc1 to arc2    
    # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
    
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc2.arc_radius        
    r2 = arc2.arc_radius    
    
    dc1c2 = np.sqrt( c_x*c_x + c_y*c_y  )
    
    psi1 = np.arctan2(c_y,c_x)
    psi2 = np.arcsin(r2/dc1c2 )
    psi3 = np.arcsin(rho/(r1+rho) )

    al1_min = psi1+psi2+psi3
    al2_min = psi1+psi2+np.pi/2
    
    if utils.InInt(arc1.angPos_lb, arc1.angPos_ub, al1_min) and utils.InInt(arc2.angPos_lb, arc2.angPos_ub, al2_min):
        phi1 = np.mod(np.pi/2-psi3, 2*np.pi)
        Ls = dc1c2*np.cos(psi2)-(r1+rho)*np.cos(psi3)
    else:
        return [np.nan, np.nan], [np.nan, np.nan]
    
    return [al1_min, al2_min], [rho*phi1, Ls]

def LocalMinRS(arc1, arc2, rho):
    # Local minimum of RS path from arc1 to arc2    
    # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
    
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc2.arc_radius        
    r2 = arc2.arc_radius    
    
    dc1c2 = np.sqrt( c_x*c_x + c_y*c_y  )
    
    psi1 = np.arctan2(c_y,c_x)
    psi2 = np.arcsin(r2/dc1c2 )
    psi3 = np.arcsin(rho/(r1-rho) )

    al1_min = psi1+psi2-psi3
    al2_min = psi1+psi2+np.pi/2
    
    if utils.InInt(arc1.angPos_lb, arc1.angPos_ub, al1_min) and utils.InInt(arc2.angPos_lb, arc2.angPos_ub, al2_min):
        phi1 = np.mod(3*np.pi/2-psi3, 2*np.pi)
        Ls = dc1c2*np.cos(psi2)-(r1-rho)*np.cos(psi3)
    else:
        return [np.nan, np.nan], [np.nan, np.nan]
    
    return [al1_min, al2_min], [rho*phi1, Ls]

def LocalMinSL(arc1, arc2, rho):
    # Local minimum of SL path from arc1 to arc2    
    # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
    
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc2.arc_radius        
    r2 = arc2.arc_radius    
    
    dc1c2 = np.sqrt( c_x*c_x + c_y*c_y  )
    
    psi1 = np.arctan2(c_y,c_x)
    psi2 = np.arcsin(r1/dc1c2 )
    psi3 = np.arcsin(rho/(r2+rho) )

    al1_min = np.pi/2+psi1-psi2
    al2_min = np.pi+psi1-psi2-psi3
    
    if utils.InInt(arc1.angPos_lb, arc1.angPos_ub, al1_min) and utils.InInt(arc2.angPos_lb, arc2.angPos_ub, al2_min):
        phi2 = np.mod(np.pi/2-psi3, 2*np.pi)
        Ls = dc1c2*np.cos(psi2)-(r2+rho)*np.cos(psi3)
    else:
        return [np.nan, np.nan], [np.nan, np.nan]
    
    return [al1_min, al2_min], [Ls, rho*phi2]

def LocalMinSR(arc1, arc2, rho):
    # Local minimum of SR path from arc1 to arc2    
    # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
    
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc2.arc_radius        
    r2 = arc2.arc_radius    
    
    dc1c2 = np.sqrt( c_x*c_x + c_y*c_y  )
    
    psi1 = np.arctan2(c_y,c_x)
    psi2 = np.arcsin(r1/dc1c2 )
    psi3 = np.arcsin(rho/(r2-rho) )

    al1_min = np.pi/2+psi1-psi2
    al2_min = np.pi+psi1-psi2+psi3
    
    if utils.InInt(arc1.angPos_lb, arc1.angPos_ub, al1_min) and utils.InInt(arc2.angPos_lb, arc2.angPos_ub, al2_min):
        phi2 = np.mod(3*np.pi/2-psi3, 2*np.pi)
        Ls = dc1c2*np.cos(psi2)-(r2-rho)*np.cos(psi3)
    else:
        return [np.nan, np.nan], [np.nan, np.nan]
    
    return [al1_min, al2_min], [Ls, rho*phi2]

def LocalMinLR(arc1, arc2, rho):
    # Local minimum of LR path from arc1 to arc2    
    # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
    
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc1.arc_radius        
    r2 = arc2.arc_radius    
    
    dc1c2 = np.sqrt( c_x**2 + c_y**2 )
    
    if r1+rho >= dc1c2 + 3*rho-r2 or dc1c2 >= r1+4*rho-r2 or 2*rho-r2>= r1+ dc1c2:
        return [np.nan, np.nan], [np.nan, np.nan]
    psi1 = np.arctan2(c_y,c_x)
    cos_psi2 = ((r1+rho)**2+dc1c2**2-(3*rho-r2)**2)/(2*dc1c2*(r1+rho))
    psi2 = np.arccos(cos_psi2)
    
    al1_min = psi1+psi2
    cos_phi1 = ((r1+rho)**2+(3*rho-r2)**2-dc1c2**2)/(2*(r1+rho)*(3*rho-r2))
    phi1 = np.arccos(cos_phi1)
    al2_min = psi1+psi2+phi1-np.pi
    
    if utils.InInt(arc1.angPos_lb, arc1.angPos_ub, al1_min) and utils.InInt(arc2.angPos_lb, arc2.angPos_ub, al2_min):
        phi2 = np.pi
    else:
        return [np.nan, np.nan], [np.nan, np.nan]
    
    return [al1_min, al2_min], [rho*phi1, rho*phi2]

def LocalMinRL(arc1, arc2, rho):
    # Local minimum of RL path from arc1 to arc2    
    # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
    
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc1.arc_radius        
    r2 = arc2.arc_radius    
    
    dc1c2 = np.sqrt( c_x**2 + c_y**2 )
    
    A = r1-rho
    B = dc1c2
    C = 3*rho+r2
    
    if r1+A >= B+C or B >= A+C or C>= A+B:
        return [np.nan, np.nan], [np.nan, np.nan]
    psi1 = np.arctan2(c_y,c_x)
    cos_psi2 = (A**2+B**2-C**2)/(2*A*B)
    psi2 = np.arccos(cos_psi2)
    
    al1_min = psi1+psi2
    cos_piminusphi1 = (A**2+C**2-B**2)/(2*A*C)
    phi1 = np.pi-np.arccos(cos_piminusphi1)
    al2_min = psi1+psi2-phi1+np.pi
    
    if utils.InInt(arc1.angPos_lb, arc1.angPos_ub, al1_min) and utils.InInt(arc2.angPos_lb, arc2.angPos_ub, al2_min):
        phi2 = np.pi
    else:
        return [np.nan, np.nan], [np.nan, np.nan]
    
    return [al1_min, al2_min], [rho*phi1, rho*phi2]

if __name__ == "__main__":

    LSL =0; LSR = 1; RSL = 2; RSR = 3; RLR = 4; LRL = 5;     
    rho = 1    
    pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
    arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
    arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')
    
    ############################# Test local min LS #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.1, 6.2)
    # arc2 = utils.Arc(-6,-2, 2.5, 0.1, 6.2)
    # minAlphasLS, segLengthsLS = LocalMinLS(arc1, arc2, rho)
    # al1 = minAlphasLS[0]
    # al2 = minAlphasLS[1]
    
    # if np.isfinite(al1):
        
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minLS = np.array([iniPt[0], iniPt[1], iniHdng])   
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minLS, 'LS', segLengthsLS, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)        
    #     plt.axis('equal')
    #     plt.show()
    
    ############################# Test local min RS #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.1, 6.2)
    # arc2 = utils.Arc(1,-6, 2.5, 0.1, 6.2)
    # minAlphasRS, segLengthsRS = LocalMinRS(arc1, arc2, rho)
    # al1 = minAlphasRS[0]
    # al2 = minAlphasRS[1]
    
    # if np.isfinite(al1):
        
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minRS = np.array([iniPt[0], iniPt[1], iniHdng])   
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minRS, 'RS', segLengthsRS, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)        
    #     plt.axis('equal')
    #     plt.show()
    
############################# Test local min SL #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.1, 6.2)
    # arc2 = utils.Arc(1,6, 2.5, 0.1, 6.2)
    # minAlphasSL, segLengthsSL = LocalMinSL(arc1, arc2, rho)
    # al1 = minAlphasSL[0]
    # al2 = minAlphasSL[1]
    
    # if np.isfinite(al1):
        
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minSL = np.array([iniPt[0], iniPt[1], iniHdng])   
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minSL, 'SL', segLengthsSL, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)        
    #     plt.axis('equal')
    #     plt.show()

############################# Test local min SR #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.01, 6.28)
    # arc2 = utils.Arc(4, -5, 2.5, 0.01, 6.28)
    # minAlphasSR, segLengthsSR = LocalMinSR(arc1, arc2, rho)
    # al1 = minAlphasSR[0]
    # al2 = minAlphasSR[1]
    
    # if np.isfinite(al1):
        
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minSR = np.array([iniPt[0], iniPt[1], iniHdng])   
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minSR, 'SR', segLengthsSR, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)        
    #     plt.axis('equal')
    #     plt.show()
        
############################# Test local min LR #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.01, 6.28)
    # arc2 = utils.Arc(-1., -3.5, 2.5, 0.01, 6.28)
    # minAlphasLR, segLengthsLR = LocalMinLR(arc1, arc2, rho)
    # al1 = minAlphasLR[0]
    # al2 = minAlphasLR[1]
    
    # if np.isfinite(al1):
        
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minLR = np.array([iniPt[0], iniPt[1], iniHdng])   
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minLR, 'LR', segLengthsLR, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)        
    #     plt.axis('equal')
    #     plt.show()
    
############################# Test local min RL #############################
    
    arc1 = utils.Arc(0,0, 2.5, 0.01, 6.28)
    arc2 = utils.Arc(4., 2.5, 2.5, 0.01, 6.28)
    minAlphasRL, segLengthsRL = LocalMinRL(arc1, arc2, rho)
    al1 = minAlphasRL[0]
    al2 = minAlphasRL[1]
    
    if np.isfinite(al1):
        
        iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
        iniHdng = al1-np.pi/2
        iniConf_minRL = np.array([iniPt[0], iniPt[1], iniHdng])   
        finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
        finHdng = al2-np.pi/2     
        du.PlotDubPathSegments(iniConf_minRL, 'RL', segLengthsRL, rho, pathfmt)
        
        utils.PlotArc(arc1, arcfmt)
        utils.PlotArc(arc2, arcfmt)        
        utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
        utils.PlotArrow(finPt, finHdng, 1, arrowfmt)        
        plt.axis('equal')
        plt.show()