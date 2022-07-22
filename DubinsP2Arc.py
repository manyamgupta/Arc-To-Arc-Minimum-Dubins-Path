import numpy as np
from numpy import pi,cos,sin
import matplotlib.pyplot as plt
import dubins
import dubutils as du 
import utils
from collections import namedtuple
from types import SimpleNamespace

def LocalMinLSL(arc, rho):
    # Local minimum of LSL path from point to an arc
    # Assumption: start config is [0,0,0], final tangent on arc in clockwise direction
    
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius    
    
    dc1ct = np.sqrt( c_x*c_x + (c_y-rho)*(c_y-rho)  )
    
    psi1 = np.arctan2(c_y-rho,c_x)
    psi2 = np.arcsin(rho/dc1ct )

    if dc1ct > rho+arcRad:               
        phi1 = psi1+psi2  
        phi2 = np.arccos(rho/(rho+arcRad))
    else:               
        phi1 = psi1-psi2+pi
        phi2 = -np.arccos(rho/(rho+arcRad))
        
    thetaMin = phi1+phi2
    minAlpha = np.mod(thetaMin+pi/2, 2*pi)
    if not utils.InInt(arc.angPos_lb, arc.angPos_ub, minAlpha):
        minAlpha = None
        lengthLSL = None
    else:
        phi1 = np.mod(phi1, 2*pi)
        phi2 = np.mod(phi2, 2*pi)
        Ls = np.sqrt(np.power(c_x + (rho+arcRad)*cos(minAlpha),2)+np.power(c_y+(rho+arcRad)*sin(minAlpha)-rho,2)  )
        lengthLSL = Ls+rho*(phi1+phi2)
    
    return minAlpha, lengthLSL

def LocalMinLSR(arc, rho):
     
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius     
    
    dc1ct = np.sqrt(c_x*c_x + (c_y-rho)*(c_y-rho) )
    psi1 = np.arctan2(c_y-rho,c_x)
    psi2 = np.arcsin(rho/np.sqrt( c_x*c_x + (c_y-rho)*(c_y-rho) ) )

    if dc1ct >= arcRad-rho:
        phi1 = np.mod(psi1+psi2, 2*pi)
        phi2 = np.mod(pi+np.arccos(rho/(arcRad-rho)), 2*pi)           
    else:
        phi1 = np.mod(psi1-psi2+pi, 2*pi)
        phi2 = np.mod(pi-np.arccos(rho/(arcRad-rho)), 2*pi)           
        
    thetaMin = phi1-phi2
    minAlpha = np.mod(thetaMin+pi/2, 2*pi)
    if not utils.InInt(arc.angPos_lb, arc.angPos_ub, minAlpha):
        minAlpha = None
        lengthLSR = None
    else:    
        dc1ct = np.sqrt( np.power(c_x+(arcRad-rho)*cos(minAlpha),2)+ np.power(c_y+(arcRad-rho)*sin(minAlpha)-rho,2))
        Ls = np.sqrt(dc1ct*dc1ct - 4*rho*rho )    
        lengthLSR = Ls + rho*(phi1+phi2)

    return minAlpha, lengthLSR

def LocalMinRSL(arc, rho):
    
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius      
    Lc1ct = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho))    
    phi2 = np.arccos(rho/(rho+arcRad))    
    psi1 = np.mod(np.arctan2(c_y+rho,c_x), 2*pi)
    psi2 = np.arcsin(rho/np.sqrt( c_x*c_x + (c_y+rho)*(c_y+rho) ) )
    
    if Lc1ct>= arcRad+rho:
        phi1 = np.mod(-psi1+psi2, 2*pi)
    else:
        phi1 = np.mod(-psi1-psi2+pi, 2*pi)
        phi2 = np.mod(-phi2, 2*pi)
    
    minAlpha = np.mod(-phi1+phi2+pi/2, 2*pi)
    if not utils.InInt(arc.angPos_lb, arc.angPos_ub, minAlpha):
        minAlpha = None
        lengthRSL = None
    else:
        Ls = np.sqrt( np.power(c_x + (rho+arcRad)*cos(minAlpha),2) + np.power(c_y + (rho+arcRad)*sin(minAlpha)+rho, 2) - 4*rho*rho )    
        lengthRSL = Ls + rho*(phi1+phi2)
    
    return minAlpha, lengthRSL

def LocalMinRSR(arc, rho):
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius    
    dc1ct = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho) )
    
    psi1 = np.arctan2(c_y+rho,c_x)
    psi2 = np.arcsin(rho/np.sqrt( c_x*c_x + (c_y+rho)*(c_y+rho) ) )
    
    if dc1ct > arcRad-rho:
        phi2_min = pi+ np.arccos(rho/(arcRad-rho))
        phi1_min = np.mod(-psi1+psi2, 2*pi)
    else:
        phi2_min = pi- np.arccos(rho/(arcRad-rho))
        phi1_min = np.mod(-psi1-psi2-pi, 2*pi)
    
    thetaMin = np.mod(-phi1_min-phi2_min, 2*pi)
    minAlpha = np.mod(thetaMin+pi/2, 2*pi)
    if not utils.InInt(arc.angPos_lb, arc.angPos_ub, minAlpha):
        minAlpha = None
        lengthRSR = None
    else:
        Ls = np.sqrt( np.power(c_x+(arcRad-rho)*cos(minAlpha),2) + np.power( (c_y+(arcRad-rho)*sin(minAlpha)+rho), 2) )    
        lengthRSR = Ls + rho*(phi1_min+phi2_min)
    
    return minAlpha, lengthRSR

def LocalMinRLR(arc, rho ):
    # computation of min alpha using analytic result
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
    l12 = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho))
    l23 = arcRad-rho    
    quarticEq = [-3,  (-16*rho*rho+8*l23*l23+8*l12*l12), -4*(l23*l23-l12*l12)*(l23*l23-l12*l12)]
    qeRoots = np.roots(quarticEq)
    
    l13a = np.sqrt(qeRoots[0])
    l13b = np.sqrt(qeRoots[1])
    
    if np.imag(l13a)==0:
        ga = (l13a*l13a-l12*l12 + l23*l23 )/(2*l13a)
        psi4a = np.arccos((l13a-ga)/l12)    
        psi2a = np.arctan2(c_y+rho,c_x)+psi4a
        
        c3xa = l13a*cos(psi2a)
        c3ya = l13a*sin(psi2a)-rho   
        alphamina = np.mod(np.arctan2( c3ya-c_y, c3xa-c_x), 2*np.pi)
    else:
        alphamina = np.nan
    
    if np.imag(l13b) == 0:
        gb = (l13b*l13b-l12*l12 + l23*l23 )/(2*l13b)
        psi4b = np.arccos((l13b-gb)/l12)    
        psi2b = np.arctan2(c_y+rho,c_x)+psi4b
        
        c3xb = l13b*cos(psi2b)
        c3yb = l13b*sin(psi2b)-rho   
        alphaminb = np.mod(np.arctan2( c3yb-c_y, c3xb-c_x), 2*np.pi)
    else:
        alphaminb = np.nan
    
    minAlphaVec = np.array([alphamina, alphaminb])
    
    lengthRLR = []
    minAlphaVecFeas = []
    PathRLR = None
    for indx, alMin in enumerate(minAlphaVec):
        # alMin = minAlpha[indx]
        if np.isfinite(alMin):  
            if utils.InInt(arc.angPos_lb, arc.angPos_ub, alMin):                
                prFinalConf = [c_x+arcRad*cos(alMin), c_y+arcRad*sin(alMin), alMin-pi/2]
                PathRLR = dubins.path([0,0,0], prFinalConf, rho, 4)
        if PathRLR is not None:
            lengthRLR.append(PathRLR.path_length())
            minAlphaVecFeas.append(alMin)
        PathRLR = None
        
    return minAlphaVecFeas, lengthRLR

def LocalMinLRL(arc, rho ):
    # computation of min alpha using analytic result
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
    
    l12 = np.sqrt(c_x*c_x + (c_y-rho)*(c_y-rho))
    l23 = arcRad+rho
    
    quarticEq = [-3,  8*(l12*l12+l23*l23-2*rho*rho), -4*(l12*l12-l23*l23)*(l12*l12-l23*l23)]
    qeRoots = np.roots(quarticEq)    
    l13a = np.sqrt(qeRoots[0])
    l13b = np.sqrt(qeRoots[1])
    
    if np.imag(l13a)==0:
        ga = (l12*l12 +l13a*l13a - l23*l23 )/(2*l13a)
        psi4a = np.arccos(ga/l12)    
        psi2a = np.arctan2(c_y-rho,c_x)-psi4a
        
        c3xa = l13a*cos(psi2a)
        c3ya = l13a*sin(psi2a)+rho   
        alphamina = np.mod(np.arctan2( c3ya-c_y, c3xa-c_x), 2*np.pi)
    else:
        alphamina = np.nan
        
    if np.imag(l13b)==0:        
        gb = (l12*l12 +l13b*l13b - l23*l23 )/(2*l13b)
        psi4b = np.arccos(gb/l12)    
        psi2b = np.arctan2(c_y-rho,c_x)-psi4b
        
        c3xb = l13b*cos(psi2b)
        c3yb = l13b*sin(psi2b)+rho   
        alphaminb = np.mod(np.arctan2( c3yb-c_y, c3xb-c_x), 2*np.pi)
    else:
        alphaminb = np.nan
    
    minAlphaVec = np.array([alphamina, alphaminb])
    
    lengthLRL = []
    minAlphaVecFeas = []
    PathLRL = None
    
    for indx, alMin in enumerate(minAlphaVec):    
        if np.isfinite(alMin): 
            if utils.InInt(arc.angPos_lb, arc.angPos_ub, alMin):             
                prFinalConf = [c_x+arcRad*cos(alMin), c_y+arcRad*sin(alMin), alMin-pi/2]
                PathLRL = dubins.path([0,0,0], prFinalConf, rho, 5)
        if PathLRL is not None:                        
            lengthLRL.append(PathLRL.path_length())
            minAlphaVecFeas.append(alMin)
        PathLRL = None

    return minAlphaVecFeas, lengthLRL

def PathLS(arc, rho):
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
    
    dist_c1ct = np.sqrt(c_x*c_x + (c_y-rho)*(c_y-rho))
    if np.abs((rho+arcRad)/(dist_c1ct)) > 1:        
        return None, None
    psi1 = np.arcsin( (rho+arcRad)/(dist_c1ct))
    psi2 = np.arctan2(c_y-rho,c_x)
    thetaLS = psi1+psi2    
    alpha_LS = np.mod(thetaLS + pi/2, 2*pi)
    if utils.InInt(arc.angPos_lb, arc.angPos_ub, alpha_LS):
        phi1 = np.mod(thetaLS, 2*pi)    
        Ls = np.sqrt(dist_c1ct*dist_c1ct - (rho+arcRad)*(rho+arcRad) )
        lengthLS = Ls + rho*phi1
    else:
        return None, None
    
    return alpha_LS, lengthLS

def PathSL(arc, rho):
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
    
    alphaSLFeas = [np.nan, np.nan]
    lengthSL = [np.nan, np.nan]
    if c_y> arcRad+2*rho or c_y < -arcRad or c_x<-(arcRad+rho):
        # feasible SL paths = 0
        return alphaSLFeas, lengthSL
        
    psi1a = np.arcsin( (c_y-rho)/(rho + arcRad) )
    psi1b = pi - psi1a
    
    phi2a = np.mod(psi1a+pi/2, 2*pi)
    phi2b = np.mod(psi1b+pi/2, 2*pi)
    
    alphaSLa = np.mod(phi2a+pi/2, 2*pi  )
    alphaSLb = np.mod(phi2b+pi/2, 2*pi  )
    
    Lsa = c_x + (rho+arcRad)*np.cos(alphaSLa )
    Lsb = c_x + (rho+arcRad)*np.cos(alphaSLb )

    phi2Vec = [phi2a, phi2b]
    alphaSLVec = [alphaSLa, alphaSLb]
    LsVec = [Lsa, Lsb]
    
    for indx, Ls in enumerate(LsVec):        
        if Ls>=0 and utils.InInt(arc.angPos_lb, arc.angPos_ub, alphaSLVec[indx]):
            lengthSL[indx] = Ls + rho*phi2Vec[indx]
            alphaSLFeas[indx] = alphaSLVec[indx]
    
    return alphaSLFeas, lengthSL

def PathRS(arc, rho):
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
    
    dist_c1ct = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho) )    
    if dist_c1ct*dist_c1ct < (arcRad-rho)*(arcRad-rho):        
        return None, None
    psi1 = np.arctan2(c_y+rho, c_x )
    psi2 = np.arcsin( (arcRad-rho)/dist_c1ct )
    
    thetaRS = psi1+psi2
    phi1RS = np.mod(-thetaRS, 2*pi)
    
    alphaRS = np.mod(thetaRS+pi/2, 2*pi)
    if utils.InInt(arc.angPos_lb, arc.angPos_ub, alphaRS):
        Ls = np.sqrt(dist_c1ct*dist_c1ct - (arcRad-rho)*(arcRad-rho)  )
        lengthRS = Ls + rho*phi1RS
    else:
        return None, None
    
    return alphaRS, lengthRS

def PathSR(arc, rho ):
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
    
    alphaSRFeas = [np.nan, np.nan]
    LengthSR = [np.nan, np.nan]
    # dist_c1ct = np.sqrt( c_x*c_x + (c_y+rho)*(c_y+rho) )
                
    psi1 = -np.arcsin( (c_y+rho)/(arcRad-rho) )
    
    phi2a = np.mod(psi1+3*pi/2, 2*pi)
    phi2b = np.mod(pi/2-psi1, 2*pi)
    
    alphaSRa = np.mod(pi-psi1, 2*pi  )
    alphaSRb = np.mod(psi1, 2*pi  )
    
    Lsa = c_x + (arcRad-rho)*np.cos(alphaSRa )
    Lsb = c_x + (arcRad-rho)*np.cos(alphaSRb )
    
    phi2Vec = [phi2a, phi2b]
    alphaSRVec = [alphaSRa, alphaSRb]
    LsVec = [Lsa, Lsb]
    for indx, Ls in enumerate(LsVec):        
        if Ls>=0 and utils.InInt(arc.angPos_lb, arc.angPos_ub, alphaSRVec[indx]):
            LengthSR[indx] = Ls + rho*phi2Vec[indx]
            alphaSRFeas[indx] = alphaSRVec[indx]
            
    return alphaSRFeas, LengthSR 

def PathLR(arc, rho):
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
          
    Lcc = np.sqrt(c_x*c_x + (c_y-rho)*(c_y-rho))  
    
    if Lcc <= 3*rho-arcRad or Lcc>arcRad+rho:
        return [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]
    
    g = (Lcc*Lcc+(arcRad-rho)*(arcRad-rho)-4*rho*rho)/(2*Lcc)
    
    psi1 = np.arctan2(c_y-rho, c_x)
    psi2 = np.arccos( (Lcc-g)/(2*rho))
    psi3 = np.arcsin(g/(arcRad-rho) ) + (pi/2-psi2)
    
    phi1a = np.mod(psi1-psi2+pi/2, 2*pi)
    phi2a = np.mod(pi+psi3, 2*pi)
    
    phi1b = np.mod(psi1+psi2+pi/2, 2*pi)
    phi2b = np.mod(pi-psi3, 2*pi)
    
    thetaLRa = np.mod(phi1a-phi2a, 2*pi)
    alphaLRa = np.mod(thetaLRa+ pi/2, 2*pi)
    distLRa = rho*(phi1a+phi2a)
    
    thetaLRb = np.mod(phi1b-phi2b, 2*pi)
    alphaLRb = np.mod(thetaLRb+ pi/2, 2*pi)
    distLRb = rho*(phi1b+phi2b)

    alphaLRVec = [alphaLRb, alphaLRa]
    distLRVec = [distLRb, distLRa]
    segLengthsVec = np.array([[rho*phi1a, rho*phi2a], [rho*phi1b, rho*phi2b]])
    for indx, alphaLR in enumerate(alphaLRVec):
        if not utils.InInt(arc.angPos_lb, arc.angPos_ub, alphaLR):
            alphaLRVec[indx] = np.nan
            distLRVec[indx] = np.nan
    return alphaLRVec, distLRVec, segLengthsVec

def PathRL(arc, rho):
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
    
    Lcc = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho))

    if Lcc>arcRad+3*rho or Lcc < arcRad-rho:
        return [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]
    
    g = (4*rho*rho + Lcc*Lcc - (arcRad+rho)*(arcRad+rho))/(2*Lcc)
    
    psi1 = np.arctan2(c_y+rho, c_x)
    psi2 = np.arccos( g/(2*rho))
    psi3 = np.arccos((Lcc-g)/(arcRad+rho) )
    
    phi1a = np.mod(-psi1+psi2+pi/2, 2*pi)
    phi2a = pi+psi2+psi3
    
    phi1b = np.mod(-psi1-psi2+pi/2, 2*pi)
    phi2b = pi-psi2-psi3
    
    thetaRLa = np.mod(-phi1a+phi2a, 2*pi)
    alphaRLa = np.mod(thetaRLa+ pi/2, 2*pi)
    distRLa = rho*(phi1a+phi2a)
    
    thetaRLb = np.mod(-phi1b+phi2b, 2*pi)
    alphaRLb = np.mod(thetaRLb+ pi/2, 2*pi)
    distRLb = rho*(phi1b+phi2b)
    
    alphaRLVec = [alphaRLb, alphaRLa]
    distRLVec = [distRLb, distRLa]
    segLengthsVec = np.array([[rho*phi1a, rho*phi2a], [rho*phi1b, rho*phi2b]])
    for indx, alphaRL in enumerate(alphaRLVec):
        if not utils.InInt(arc.angPos_lb, arc.angPos_ub, alphaRL):
            alphaRLVec[indx] = np.nan
            distRLVec[indx] = np.nan
    return alphaRLVec, distRLVec, segLengthsVec

def RLRFeasLimits(arc, rho):
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
    
    k1 = (-rho-c_y)/c_x
    k2 = (c_x*c_x + c_y*c_y + 15*rho*rho - (arcRad-rho)*(arcRad-rho))/(2*c_x)
    
    polyY = [k1*k1+1, 2*k1*k2+2*rho, k2*k2-15*rho*rho]
    yroots = np.roots(polyY)
    
    y1 = yroots[0]; y2 = yroots[1];
    if np.imag(y1) == 0:
        x1 = k1*y1+k2
        x2 = k1*y2+k2
        allimit1 = np.mod(np.arctan2(y1-c_y, x1-c_x), 2*pi)
        allimit2 = np.mod(np.arctan2(y2-c_y, x2-c_x), 2*pi)
    else:
        return [np.nan, np.nan], [np.nan, np.nan]       
 
    alLimits = [allimit1, allimit2]
    alLimitsFeas = []
    lengthRLR = []
    for indx, al in enumerate(alLimits):
        if np.isfinite(al) and utils.InInt(arc.angPos_lb, arc.angPos_ub, al):        
            prFinalConf = [c_x+arcRad*cos(al), c_y+arcRad*sin(al), al-pi/2]
            PathRLR = dubins.path([0,0,0], prFinalConf, rho, 4)
            if str(PathRLR) == 'None':
                ale= al-0.0001
                prFinalConf = [c_x+arcRad*cos(ale), c_y+arcRad*sin(ale), ale-pi/2]
                PathRLR = dubins.path([0,0,0], prFinalConf, rho, 4)
            if str(PathRLR) == 'None':            
                ale= al+0.0001
                prFinalConf = [c_x+arcRad*cos(ale), c_y+arcRad*sin(ale), ale-pi/2]
                PathRLR = dubins.path([0,0,0], prFinalConf, rho, 4)
            if str(PathRLR) != 'None':
                alLimitsFeas.append(al)
                lengthRLR.append(PathRLR.path_length())                
    
    
    return alLimitsFeas, lengthRLR

def LRLFeasLimits(arc, rho,):
    c_x = arc.c_x
    c_y = arc.c_y
    arcRad = arc.arc_radius 
    
    k1 = (rho-c_y)/c_x
    k2 = (c_x*c_x + c_y*c_y + 15*rho*rho - (arcRad+rho)*(arcRad+rho))/(2*c_x)
    
    polyY = [k1*k1+1, 2*k1*k2-2*rho, k2*k2-15*rho*rho]
    
    yroots = np.roots(polyY)
    
    y1 = yroots[0]; y2 = yroots[1];
    
    if np.imag(y1) == 0:
        x1 = k1*y1+k2
        x2 = k1*y2+k2
    
        allimit1 = np.mod(np.arctan2(y1-c_y, x1-c_x), 2*pi)
        allimit2 = np.mod(np.arctan2(y2-c_y, x2-c_x), 2*pi)
    else:
        return [np.nan, np.nan], [np.nan, np.nan]
    
    alLimits = [allimit1, allimit2]
    lengthLRL = []
    alLimitsFeas = []
    for indx, al in enumerate(alLimits):
        
        if np.isfinite(al) and utils.InInt(arc.angPos_lb, arc.angPos_ub, al):                
            prFinalConf = [c_x+arcRad*cos(al), c_y+arcRad*sin(al), al-pi/2]
            PathLRL = dubins.path([0,0,0], prFinalConf, rho, 5)
            if str(PathLRL) == 'None':
                ale= al-0.00001
                prFinalConf = [c_x+arcRad*cos(ale), c_y+arcRad*sin(ale), ale-pi/2]
                PathLRL = dubins.path([0,0,0], prFinalConf, rho, 5)
            if str(PathLRL) == 'None':
                ale= al+0.00001
                prFinalConf = [c_x+arcRad*cos(ale), c_y+arcRad*sin(ale), ale-pi/2]
                PathLRL = dubins.path([0,0,0], prFinalConf, rho, 5)
            if str(PathLRL) != 'None':
                alLimitsFeas.append(al)            
                lengthLRL.append(PathLRL.path_length())

    return alLimitsFeas, lengthLRL

if __name__ == "__main__":

    LSL =0; LSR = 1; RSL = 2; RSR = 3; RLR = 4; LRL = 5; 
    iniConf = np.array([0,0,0])
    rho = 1    
    pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
    arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
    arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')
    
    ############################# Test LSL #############################
    
    # arc1 = utils.Arc(-5,6, 2.5, -1.5, 1)
    # minAlphaLSL, lengthLSL = LocalMinLSL(arc1, rho)
    # if minAlphaLSL:
        
    #     finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(minAlphaLSL), arc1.c_y+arc1.arc_radius*np.sin(minAlphaLSL)])
    #     finHdng = minAlphaLSL-np.pi/2
    #     finConf_minLSL = np.array([finPt[0], finPt[1], finHdng])
    #     path_minLSL = dubins.path(iniConf, finConf_minLSL, rho, LSL)
    #     du.PlotDubinsPath(path_minLSL, pathfmt)
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    #     plt.axis('equal')
    #     plt.show()
    
    ############################# Test LSR #############################
    
    # arc1 = utils.Arc(5,-6, 2.5, .5, 5)
    # minAlphaLSR, lengthLSR = LocalMinLSR(arc1, rho)
    # if minAlphaLSR:
        
    #     finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(minAlphaLSR), arc1.c_y+arc1.arc_radius*np.sin(minAlphaLSR)])
    #     finHdng = minAlphaLSR-np.pi/2
    #     finConf_minLSR = np.array([finPt[0], finPt[1], finHdng])
    #     path_minLSR = dubins.path(iniConf, finConf_minLSR, rho, LSR)
    #     du.PlotDubinsPath(path_minLSR, pathfmt)
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    #     plt.axis('equal')
    #     plt.show()

############################# Test RSL #############################
    
    # arc1 = utils.Arc(5, 6, 2.5, 1.5, 4.5)
    # minAlphaRSL, lengthRSL = LocalMinRSL(arc1, rho)
    # if minAlphaRSL:        
    #     finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(minAlphaRSL), arc1.c_y+arc1.arc_radius*np.sin(minAlphaRSL)])
    #     finHdng = minAlphaRSL-np.pi/2
    #     finConf_minRSL = np.array([finPt[0], finPt[1], finHdng])
    #     path_minRSL = dubins.path(iniConf, finConf_minRSL, rho, RSL)
    #     du.PlotDubinsPath(path_minRSL, pathfmt)
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    #     plt.axis('equal')
    #     plt.show()
    
############################# Test RSR #############################
    
    # arc1 = utils.Arc(5, 6, 2.5, 0.1, 4.)
    # minAlphaRSR, lengthRSR = LocalMinRSR(arc1, rho)
    # if minAlphaRSR:        
    #     finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(minAlphaRSR), arc1.c_y+arc1.arc_radius*np.sin(minAlphaRSR)])
    #     finHdng = minAlphaRSR-np.pi/2
    #     finConf_minRSR = np.array([finPt[0], finPt[1], finHdng])
    #     path_minRSR = dubins.path(iniConf, finConf_minRSR, rho, RSR)
    #     du.PlotDubinsPath(path_minRSR, pathfmt)
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    #     plt.axis('equal')
    #     plt.show()
    
############################# Test RLR #############################
    
    # arc1 = utils.Arc(-1.5, -2., 2.5, .1, 6.)
    # minAlphaRLR_vec, lengthRLR_vec = LocalMinRLR(arc1, rho)
    # print(f"{minAlphaRLR_vec=} {lengthRLR_vec=}")
    # for indx, minAlphaRLR in enumerate(minAlphaRLR_vec):
    #     if np.isfinite(minAlphaRLR):
    #         finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(minAlphaRLR), arc1.c_y+arc1.arc_radius*np.sin(minAlphaRLR)])
    #         finHdng = minAlphaRLR-np.pi/2
    #         finConf_minRLR = np.array([finPt[0], finPt[1], finHdng])
    #         path_minRLR = dubins.path(iniConf, finConf_minRLR, rho, RLR)
    #         du.PlotDubinsPath(path_minRLR, pathfmt)
    #         utils.PlotArc(arc1, arcfmt)
    #         utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    # plt.axis('equal')
    # plt.show()
    
############################ Test RLR #############################
    
    # arc1 = utils.Arc(6.5, 4., 2.5, .1, 6.)
    # minAlphaLRL_vec, lengthLRL_vec = LocalMinLRL(arc1, rho)
    # print(f"{minAlphaLRL_vec=} {lengthLRL_vec=}")
    # for indx, minAlphaLRL in enumerate(minAlphaLRL_vec):
    #     if np.isfinite(minAlphaLRL):
    #         finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(minAlphaLRL), arc1.c_y+arc1.arc_radius*np.sin(minAlphaLRL)])
    #         finHdng = minAlphaLRL-np.pi/2
    #         finConf_minLRL = np.array([finPt[0], finPt[1], finHdng])
    #         path_minLRL = dubins.path(iniConf, finConf_minLRL, rho, LRL)
    #         du.PlotDubinsPath(path_minLRL, pathfmt)
    #         utils.PlotArc(arc1, arcfmt)
    #         utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    # plt.axis('equal')
    # plt.show()
    
############################# Test LS #############################
    
    # arc1 = utils.Arc(5, 6, 2.5, 0.1, 2.)
    # alphaLS, lengthLS = PathLS(arc1, rho)
    # if alphaLS:        
    #     finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(alphaLS), arc1.c_y+arc1.arc_radius*np.sin(alphaLS)])
    #     finHdng = alphaLS-np.pi/2
    #     finConf_LS = np.array([finPt[0], finPt[1], finHdng])
    #     path_LS = dubins.path(iniConf, finConf_LS, rho, LSL)
    #     du.PlotDubinsPath(path_LS, pathfmt)
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    #     plt.axis('equal')
    #     plt.show()
    
############################# Test SL #############################
    
    # arc1 = utils.Arc(4, 3, 2.5, 0.1, 3.)
    # alphaSLVec, lengthSLVec = PathSL(arc1, rho)
    # print(f"{alphaSLVec=}")
    # for alphaSL in alphaSLVec:
    #     if np.isfinite(alphaSL):
    #         finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(alphaSL), arc1.c_y+arc1.arc_radius*np.sin(alphaSL)])
    #         finHdng = alphaSL-np.pi/2
    #         finConf_SL = np.array([finPt[0], finPt[1], finHdng])
    #         path_SL = dubins.path(iniConf, finConf_SL, rho, LSL)
    #         du.PlotDubinsPath(path_SL, pathfmt)
    #         utils.PlotArc(arc1, arcfmt)
    #         utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    # plt.axis('equal')
    # plt.show()
    
############################# Test RS #############################
    
    # arc1 = utils.Arc(-3, -2, 2.5, 0.1, 6.)
    # alphaRS, lengthRS = PathRS(arc1, rho)
    # if alphaRS:        
    #     finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(alphaRS), arc1.c_y+arc1.arc_radius*np.sin(alphaRS)])
    #     finHdng = alphaRS-np.pi/2
    #     finConf_RS = np.array([finPt[0], finPt[1], finHdng])
    #     path_RS = dubins.path(iniConf, finConf_RS, rho, RSL)
    #     du.PlotDubinsPath(path_RS, pathfmt)
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    #     plt.axis('equal')
    #     plt.show()
    
############################# Test SR #############################
    
    # arc1 = utils.Arc(3, 0, 2.5, 4., 6.)
    # alphaSRVec, lengthSRVec = PathSR(arc1, rho)
    # print(f"{alphaSRVec=}")
    # for alphaSR in alphaSRVec:
    #     if np.isfinite(alphaSR):
    #         finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(alphaSR), arc1.c_y+arc1.arc_radius*np.sin(alphaSR)])
    #         finHdng = alphaSR-np.pi/2
    #         finConf_SR = np.array([finPt[0], finPt[1], finHdng])
    #         path_SR = dubins.path(iniConf, finConf_SR, rho, LSR)
    #         du.PlotDubinsPath(path_SR, pathfmt)
    #         utils.PlotArc(arc1, arcfmt)
    #         utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    # plt.axis('equal')
    # plt.show()
    
############################# Test LR #############################
    
    # arc1 = utils.Arc(-3, 2, 2.5, 0.1, 6.)
    # alphaLRVec, lengthLRVec, segLengthsVec = PathLR(arc1, rho)
    # print(f"{alphaLRVec=}")
    # for indx, alphaLR in enumerate(alphaLRVec):
    #     if np.isfinite(alphaLR):
    #         finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(alphaLR), arc1.c_y+arc1.arc_radius*np.sin(alphaLR)])
    #         finHdng = alphaLR-np.pi/2
    #         finConf_LR = np.array([finPt[0], finPt[1], finHdng])
    #       # path_LR = dubins.path(iniConf, finConf_LR, rho, LRL)
    #         # du.PlotDubinsPath(path_LR, pathfmt)
    #         du.PlotDubPathSegments(iniConf, 'LR', segLengthsVec[indx], rho, pathfmt)
    #         utils.PlotArc(arc1, arcfmt)
    #         utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    # plt.axis('equal')
    # plt.show()
    
############################# Test RL #############################
    
    # arc1 = utils.Arc(4, 2, 2.5, 0.1, 6.)
    # alphaRLVec, lengthRLVec, segLengthsVec = PathRL(arc1, rho)
    # print(f"{alphaRLVec=}")
    # for indx, alphaRL in enumerate(alphaRLVec):
    #     if np.isfinite(alphaRL):
    #         finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(alphaRL), arc1.c_y+arc1.arc_radius*np.sin(alphaRL)])
    #         finHdng = alphaRL-np.pi/2
    #         # finConf_RL = np.array([finPt[0], finPt[1], finHdng])
    #         # path_RL = dubins.path(iniConf, finConf_RL, rho, LRL)
    #         # du.PlotDubinsPath(path_LR, pathfmt)
    #         du.PlotDubPathSegments(iniConf, 'RL', segLengthsVec[indx], rho, pathfmt)
    #         utils.PlotArc(arc1, arcfmt)
    #         utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    # plt.axis('equal')
    # plt.show()
    
############################# Test RLR/LRL Feasibility #############################
    
    arc1 = utils.Arc(4, 2, 2.5, 0.1, 6.)
    alphaRLRFeasLimits, lengthRLRVec = RLRFeasLimits(arc1, rho)
    # alphaRLRFeasLimits, lengthRLRVec = LRLFeasLimits(arc1, rho)
    print(f"{alphaRLRFeasLimits=}")
    for indx, alphaRLR in enumerate(alphaRLRFeasLimits):
        if np.isfinite(alphaRLR):
            finPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(alphaRLR), arc1.c_y+arc1.arc_radius*np.sin(alphaRLR)])
            finHdng = alphaRLR-np.pi/2
            finConf_RLR = np.array([finPt[0], finPt[1], finHdng])
            path_RLR = dubins.path(iniConf, finConf_RLR, rho, RLR)
            du.PlotDubinsPath(path_RLR, pathfmt)            
            utils.PlotArc(arc1, arcfmt)
            utils.PlotArrow(finPt, finHdng, 1, arrowfmt)
    plt.axis('equal')
    plt.show()