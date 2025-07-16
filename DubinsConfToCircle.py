from cmath import isfinite
import numpy as np
from numpy import pi,cos,sin
import matplotlib.pyplot as plt
import dubins
import dubutils as du 
import utils
from types import SimpleNamespace
from dataclasses import dataclass

@dataclass
class CandidateP2CPath:
    pathType: str 
    angPos: float # angular position at the arc
    segLengths: tuple #segment lengths

    
class P2CircleDubins:
    def __init__(self, circle_cntr, circle_radius, rho):     

        # Assumption: start config is (0,0,0)        
        self.tcircle = utils.Circle(circle_cntr[0], circle_cntr[1], circle_radius)
        self.rho = rho
        
        return
            
    def LocalMinLSL(self):
        # Local minimum of LSL path from point to an Circle
        # Assumption: start config is [0,0,0], final tangent on arc in clockwise direction
        
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius    
        rho = self.rho
        
        dc1ct = np.sqrt( c_x*c_x + (c_y-rho)*(c_y-rho)  )
        if dc1ct<rho:
            return None, None, (None, None, None)
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

        phi1 = np.mod(phi1, 2*pi)
        phi2 = np.mod(phi2, 2*pi)
        Ls = np.sqrt(np.power(c_x + (rho+arcRad)*cos(minAlpha),2)+np.power(c_y+(rho+arcRad)*sin(minAlpha)-rho,2)  )
        lengthLSL = Ls+rho*(phi1+phi2)
        segLengths = (rho*phi1, Ls, rho*phi2)
        
        return minAlpha, lengthLSL, segLengths

    def LocalMinLSR(self):
        
        if self.tcircle.circle_radius <= 2*self.rho:
            return None, None, (None, None, None)
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius     
        rho = self.rho
        
        dc1ct = np.sqrt(c_x*c_x + (c_y-rho)*(c_y-rho) )
        if dc1ct < rho:
            return None, None, (None, None, None)
        psi1 = np.arctan2(c_y-rho,c_x)
        psi2 = np.arcsin(rho/dc1ct )

        if dc1ct >= arcRad-rho:
            phi1 = np.mod(psi1+psi2, 2*pi)
            phi2 = np.mod(pi+np.arccos(rho/(arcRad-rho)), 2*pi)           
        else:
            phi1 = np.mod(psi1-psi2+pi, 2*pi)
            phi2 = np.mod(pi-np.arccos(rho/(arcRad-rho)), 2*pi)           
            
        thetaMin = phi1-phi2
        minAlpha = np.mod(thetaMin+pi/2, 2*pi)
 
        dc1ct = np.sqrt( np.power(c_x+(arcRad-rho)*cos(minAlpha),2)+ np.power(c_y+(arcRad-rho)*sin(minAlpha)-rho,2))
        Ls = np.sqrt(dc1ct*dc1ct - 4*rho*rho )    
        lengthLSR = Ls + rho*(phi1+phi2)
        segLengths = (rho*phi1, Ls, rho*phi2)            

        return minAlpha, lengthLSR, segLengths

    def LocalMinRSL(self):
        
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius     
        rho = self.rho
        
        Lc1ct = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho))  
        if Lc1ct < rho:
            return None, None, (None, None, None)
        phi2 = np.arccos(rho/(rho+arcRad))    
        psi1 = np.mod(np.arctan2(c_y+rho,c_x), 2*pi)
        # expr = rho/np.sqrt( c_x*c_x + (c_y+rho)*(c_y+rho) )        
    
        psi2 = np.arcsin(rho/Lc1ct )
        
        if Lc1ct>= arcRad+rho:
            phi1 = np.mod(-psi1+psi2, 2*pi)
        else:
            phi1 = np.mod(-psi1-psi2+pi, 2*pi)
            phi2 = np.mod(-phi2, 2*pi)
        
        minAlpha = np.mod(-phi1+phi2+pi/2, 2*pi)
        Ls = np.sqrt( np.power(c_x + (rho+arcRad)*cos(minAlpha),2) + np.power(c_y + (rho+arcRad)*sin(minAlpha)+rho, 2) - 4*rho*rho )    
        lengthRSL = Ls + rho*(phi1+phi2)
        segLengths = (rho*phi1, Ls, rho*phi2)
        
        return minAlpha, lengthRSL, segLengths

    def LocalMinRSR(self):
        if self.tcircle.circle_radius <= 2*self.rho:
            return None, None, (None, None, None)        
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius  
        rho = self.rho
         
        dc1ct = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho) )
        
        if dc1ct < rho:
            return None, None, (None, None, None) 
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

        Ls = np.sqrt( np.power(c_x+(arcRad-rho)*cos(minAlpha),2) + np.power( (c_y+(arcRad-rho)*sin(minAlpha)+rho), 2) )    
        lengthRSR = Ls + rho*(phi1_min+phi2_min)
        segLengths = (rho*phi1_min, Ls, rho*phi2_min)
        return minAlpha, lengthRSR, segLengths

    def LocalMinRLR(self):
        # computation of min alpha using analytic result
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius  
        rho = self.rho
        
        l12 = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho))
        l23 = arcRad-rho    
        quarticEq = [-3,  (-16*rho*rho+8*l23*l23+8*l12*l12), -4*(l23*l23-l12*l12)*(l23*l23-l12*l12)]
        qeRoots = np.roots(quarticEq)
        lengthRLR = 100000000
        alMin = None
        segLengths = (None, None, None)
        for qe in qeRoots:
            if qe<=0 or np.imag(qe) !=0:
                continue
            l13 = np.sqrt(qe)
            if np.imag(l13)==0:
                g = (l13*l13-l12*l12 + l23*l23 )/(2*l13)
                if np.abs((l13-g)/l12) > 1 or np.abs(l13/4/rho)>1:
                    continue
                    # return None, None, (None, None, None)
                psi4 = np.arccos((l13-g)/l12)    
                psi2 = np.arctan2(c_y+rho,c_x)+psi4
                psi1 = np.arccos(l13/4/rho)                
                
                c3x = l13*cos(psi2)
                c3y = l13*sin(psi2)-rho   
                alphamin = np.mod(np.arctan2( c3y-c_y, c3x-c_x), 2*np.pi)
              
                # prFinalConf = [c_x+arcRad*cos(alphamin), c_y+arcRad*sin(alphamin), alphamin-pi/2]
                # PathRLR = dubins.path([0,0,0], prFinalConf, rho, 4)
                phi1 = np.mod(np.pi/2-psi2+psi1, 2*np.pi)
                phi2 = np.mod(np.pi+2*psi1, 2*np.pi)
                phi3 = np.mod(-alphamin-phi1+phi2+np.pi/2, 2*np.pi)
                # if PathRLR is not None:
                    # if PathRLR.path_length() < lengthRLR:
                if rho*(phi1+phi2+phi3) < lengthRLR:                        
                    lengthRLR = rho*(phi1+phi2+phi3)
                    alMin = alphamin
                    segLengths = (rho*phi1, rho*phi2, rho*phi3)
                    # segLengths = (PathRLR.segment_length(0), PathRLR.segment_length(1), PathRLR.segment_length(2))

        return alMin, lengthRLR, segLengths 

    def LocalMinLRL(self):
        # computation of min alpha using analytic result
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius  
        rho = self.rho
        
        l12 = np.sqrt(c_x*c_x + (c_y-rho)*(c_y-rho))
        l23 = arcRad+rho
        
        quarticEq = [-3,  8*(l12*l12+l23*l23-2*rho*rho), -4*(l12*l12-l23*l23)*(l12*l12-l23*l23)]
        qeRoots = np.roots(quarticEq)   
        alMin = None
        lengthLRL = 1000000
        segLengths = (None, None, None)
        for qe in qeRoots: 
            l13 = np.sqrt(qe)
            if np.imag(l13)==0:
                g = (l12*l12 +l13*l13 - l23*l23 )/(2*l13)
                if np.abs(g/l12) >1 or np.abs(l13/4/rho) >1:
                    continue
                    # return None, None, (None, None, None)
                psi4 = np.arccos(g/l12)    
                psi2 = np.arctan2(c_y-rho,c_x)-psi4
                psi1 = np.arccos(l13/4/rho)
                c3x = l13*cos(psi2)
                c3y = l13*sin(psi2)+rho   
                alphamin = np.mod(np.arctan2( c3y-c_y, c3x-c_x), 2*np.pi)
                            
                # prFinalConf = [c_x+arcRad*cos(alphamin), c_y+arcRad*sin(alphamin), alphamin-pi/2]
                # PathLRL = dubins.path([0,0,0], prFinalConf, rho, 5)
                phi1 = np.mod(np.pi/2+psi2+psi1, 2*np.pi)
                phi2 = np.mod(np.pi+2*psi1, 2*np.pi)
                phi3 = np.mod(alphamin-phi1+phi2-np.pi/2, 2*np.pi)
                
                # if PathLRL is not None:   
                # if PathLRL.path_length() < lengthLRL:              
                if rho*(phi1+phi2+phi3) < lengthLRL:                              
                    # lengthLRL = PathLRL.path_length()
                    lengthLRL = rho*(phi1+phi2+phi3)
                    alMin = alphamin
                    # segLengths = (PathLRL.segment_length(0), PathLRL.segment_length(1), PathLRL.segment_length(2))
                    segLengths = (rho*phi1, rho*phi2, rho*phi3)
                            
        return alMin, lengthLRL, segLengths

    def PathLS(self):
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius 
        rho = self.rho
        
        dist_c1ct = np.sqrt(c_x*c_x + (c_y-rho)*(c_y-rho))
        if np.abs((rho+arcRad)/(dist_c1ct)) > 1:        
            return None, None, None
        psi1 = np.arcsin( (rho+arcRad)/(dist_c1ct))
        psi2 = np.arctan2(c_y-rho,c_x)
        thetaLS = psi1+psi2    
        alpha_LS = np.mod(thetaLS + pi/2, 2*pi)
        
        phi1 = np.mod(thetaLS, 2*pi)    
        Ls = np.sqrt(dist_c1ct*dist_c1ct - (rho+arcRad)*(rho+arcRad) )
        lengthLS = Ls + rho*phi1
        segLengths = (rho*phi1, Ls)
        
        return alpha_LS, lengthLS, segLengths

    def PathSL(self):
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius 
        rho = self.rho
        
        if c_y> arcRad+2*rho or c_y < -arcRad or c_x<-(arcRad+rho):
            # feasible SL paths = 0
            return None, None, (None, None)
            
        psi1a = np.arcsin( (c_y-rho)/(rho + arcRad) )
        # psi1b = pi - psi1a
        
        psi1_vec = (psi1a, np.pi - psi1a)
        lengthSL = 10000000
        alphaSL = None
        segLengths = (None, None)
        for psi1 in psi1_vec:
            phi2 = np.mod(psi1+pi/2, 2*pi)        
            alphaSL_cand = np.mod(phi2+pi/2, 2*pi  )        
            Ls = c_x + (rho+arcRad)*np.cos(alphaSL_cand )
            if Ls>=0:
                if Ls + rho*phi2 < lengthSL:
                    lengthSL = Ls + rho*phi2
                    alphaSL = alphaSL_cand
                    segLengths = (Ls, rho*phi2)
                
        return alphaSL, lengthSL, segLengths


    def PathRS(self):
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius 
        rho = self.rho
        
        dist_c1ct = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho) )    
        if dist_c1ct*dist_c1ct < (arcRad-rho)*(arcRad-rho):        
            return None, None, None
        psi1 = np.arctan2(c_y+rho, c_x )
        psi2 = np.arcsin( (arcRad-rho)/dist_c1ct )
        
        thetaRS = psi1+psi2
        phi1RS = np.mod(-thetaRS, 2*pi)
        
        alphaRS = np.mod(thetaRS+pi/2, 2*pi)        
        Ls = np.sqrt(dist_c1ct*dist_c1ct - (arcRad-rho)*(arcRad-rho)  )
        lengthRS = Ls + rho*phi1RS
        segLengths = (rho*phi1RS, Ls)
    
        return alphaRS, lengthRS, segLengths

    def PathSR(self):
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius 
        rho = self.rho
        
        if c_y < -arcRad or c_y > arcRad-2*rho or c_x<-(arcRad+rho):
            return None, None, (None, None)
        psi1 = -np.arcsin( (c_y+rho)/(arcRad-rho) )        
        phi2a = np.mod(psi1+3*pi/2, 2*pi)
        phi2b = np.mod(np.pi/2-psi1, 2*pi)
        
        alphaSR = None
        lengthSR = 1000000
        segLengths = (None, None)
        alphas = (np.mod(pi-psi1, 2*pi ), np.mod(psi1, 2*pi) )
        for indx, phi2 in enumerate([phi2a, phi2b]):
        
            # alphaSR_cand = np.mod(pi-psi1, 2*pi  )
            # alphaSRb = np.mod(psi1, 2*pi  )
            alphaSR_cand = alphas[indx]
        
            Ls = c_x + (arcRad-rho)*np.cos(alphaSR_cand)
            # Lsb = c_x + (arcRad-rho)*np.cos(alphaSRb )
            if Ls>=0:
                if Ls + rho*phi2 < lengthSR:
                    lengthSR = Ls + rho*phi2
                    alphaSR = alphaSR_cand
                    segLengths = (Ls, rho*phi2)
        return alphaSR, lengthSR, segLengths

    def PathLR(self):
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius 
        rho = self.rho
            
        Lcc = np.sqrt(c_x*c_x + (c_y-rho)*(c_y-rho))  
        
        if Lcc <= 3*rho-arcRad or Lcc>arcRad+rho:
            return None, None, (None, None)
        
        g = (Lcc*Lcc+(arcRad-rho)*(arcRad-rho)-4*rho*rho)/(2*Lcc)
        
        psi1 = np.arctan2(c_y-rho, c_x)
        psi2 = np.arccos( (Lcc-g)/(2*rho))
        psi3 = np.arcsin(g/(arcRad-rho) ) + (pi/2-psi2)
        phi1Vec = (np.mod(psi1-psi2+pi/2, 2*pi), np.mod(psi1+psi2+pi/2, 2*pi))
        phi2Vec = (np.mod(pi+psi3, 2*pi), np.mod(pi-psi3, 2*pi))
        lengthLR  = 100000000
        alphaLR = None
        segLengths = (None, None)
        for indx, phi1 in enumerate(phi1Vec):
            phi2 = phi2Vec[indx]
            thetaLR = np.mod(phi1-phi2, 2*pi)
            alphaLR_cand = np.mod(thetaLR+ pi/2, 2*pi)               
            
            if rho*(phi1+phi2) < lengthLR:
                lengthLR = rho*(phi1+phi2) 
                alphaLR = alphaLR_cand 
                segLengths = (rho*phi1, rho*phi2)   
                    
        return alphaLR, lengthLR, segLengths

    def PathRL(self):
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius 
        rho = self.rho
        
        Lcc = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho))

        if Lcc>arcRad+3*rho or Lcc < arcRad-rho:
            return None, None, (None, None)
        
        g = (4*rho*rho + Lcc*Lcc - (arcRad+rho)*(arcRad+rho))/(2*Lcc)
        psi1 = np.arctan2(c_y+rho, c_x)
        psi2 = np.arccos( g/(2*rho))
        psi3 = np.arccos((Lcc-g)/(arcRad+rho) )
        
        phi1Vec = (np.mod(-psi1+psi2+pi/2, 2*pi), np.mod(-psi1-psi2+pi/2, 2*pi))
        phi2Vec = (pi+psi2+psi3, pi-psi2-psi3)
        lengthRL = 10000000
        alphaRL = None
        segLengths = (None, None)
        for indx, phi1 in enumerate(phi1Vec):
            phi2 = phi2Vec[indx]
            thetaRL = np.mod(-phi1+phi2, 2*pi)
            alphaRL_cand = np.mod(thetaRL+ pi/2, 2*pi)        
              
            if rho*(phi1+phi2) < lengthRL:        
                lengthRL = rho*(phi1+phi2)
                alphaRL = alphaRL_cand 
                segLengths = (rho*phi1, rho*phi2)

        return alphaRL, lengthRL, segLengths

    def RLRFeasLimits(self):
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius 
        rho = self.rho
        
        k1 = (-rho-c_y)/c_x
        k2 = (c_x*c_x + c_y*c_y + 15*rho*rho - (arcRad-rho)*(arcRad-rho))/(2*c_x)
        
        polyY = [k1*k1+1, 2*k1*k2+2*rho, k2*k2-15*rho*rho]
        yroots = np.roots(polyY)
        
        y1 = yroots[0]; y2 = yroots[1]
        lengthRLR = 1000000
        alLimitsFeas = None
        segLengths = (None, None, None)
        for y in yroots:
            if np.imag(y) == 0:
                x = k1*y+k2
                allimit = np.mod(np.arctan2(y-c_y, x-c_x), 2*pi)
                ale = allimit
                if np.isfinite(allimit):        
                    prFinalConf = [c_x+arcRad*cos(allimit), c_y+arcRad*sin(allimit), allimit-pi/2]
                    PathRLR = dubins.path([0,0,0], prFinalConf, rho, 4)
                    if str(PathRLR) == 'None':
                        ale= allimit-0.0001
                        prFinalConf = [c_x+arcRad*cos(ale), c_y+arcRad*sin(ale), ale-pi/2]
                        PathRLR = dubins.path([0,0,0], prFinalConf, rho, 4)
                    if str(PathRLR) == 'None':            
                        ale= allimit+0.0001
                        prFinalConf = [c_x+arcRad*cos(ale), c_y+arcRad*sin(ale), ale-pi/2]
                        PathRLR = dubins.path([0,0,0], prFinalConf, rho, 4)
                    if str(PathRLR) != 'None':
                        if PathRLR.path_length()<lengthRLR:
                            alLimitsFeas = ale
                            lengthRLR = PathRLR.path_length()
                
        return alLimitsFeas, lengthRLR


    def LRLFeasLimits(self):
        c_x = self.tcircle.cntr_x
        c_y = self.tcircle.cntr_y
        arcRad = self.tcircle.circle_radius 
        rho =self.rho
        
        k1 = (rho-c_y)/c_x
        k2 = (c_x*c_x + c_y*c_y + 15*rho*rho - (arcRad+rho)*(arcRad+rho))/(2*c_x)
        
        polyY = [k1*k1+1, 2*k1*k2-2*rho, k2*k2-15*rho*rho]
        
        yroots = np.roots(polyY)
        alLimitFeas = None
        lengthLRL = 10000000
        for y in yroots:        
            if np.imag(y) == 0:
                x = k1*y+k2            
                allimit = np.mod(np.arctan2(y-c_y, x-c_x), 2*pi)   
                ale = allimit         
                if np.isfinite(allimit):                
                    prFinalConf = [c_x+arcRad*cos(allimit), c_y+arcRad*sin(allimit), allimit-pi/2]
                    PathLRL = dubins.path([0,0,0], prFinalConf, rho, 5)
                    if str(PathLRL) == 'None':
                        ale= allimit-0.00001
                        prFinalConf = [c_x+arcRad*cos(ale), c_y+arcRad*sin(ale), ale-pi/2]
                        PathLRL = dubins.path([0,0,0], prFinalConf, rho, 5)
                    if str(PathLRL) == 'None':
                        ale= allimit+0.00001
                        prFinalConf = [c_x+arcRad*cos(ale), c_y+arcRad*sin(ale), ale-pi/2]
                        PathLRL = dubins.path([0,0,0], prFinalConf, rho, 5)
                    if str(PathLRL) != 'None':
                        if PathLRL.path_length() < lengthLRL:
                            alLimitFeas = ale
                            lengthLRL = PathLRL.path_length()

        return alLimitFeas, lengthLRL    
    def PlotDubPath(self, alpha, segLengths, pathType):
        
        pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
        arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
        arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')
        arrowLength = self.rho
        finPt = np.array([self.tcircle.cntr_x+self.tcircle.circle_radius*np.cos(alpha), self.tcircle.cntr_y+self.tcircle.circle_radius*np.sin(alpha)])
        finHdng = alpha-np.pi/2
        # finConf = np.array([finPt[0], finPt[1], finHdng])
        # dubPath = dubins.path(iniConf, finConf, self.rho, du.DubPathStr2Num(pathType))
        # segLengths = (dubPath.segment_length(0), dubPath.segment_length(1), dubPath.segment_length(2))
        du.PlotDubPathSegments([0,0,0], pathType, segLengths, self.rho, pathfmt)
        
        utils.PlotCircle2(self.tcircle, arcfmt) 
        utils.PlotArrow(finPt, finHdng, arrowLength, arrowfmt) 
        utils.PlotArrow([0,0], 0, arrowLength, arrowfmt) 

        plt.axis('equal')
        plt.show()
        return
    
    def PlotDubPath2(self, alpha, pathType):
        
        pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
        arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
        arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')
        finPt = np.array([self.tcircle.cntr_x+self.tcircle.circle_radius*np.cos(alpha), self.tcircle.cntr_y+self.tcircle.circle_radius*np.sin(alpha)])
        finHdng = alpha-np.pi/2
        finConf = np.array([finPt[0], finPt[1], finHdng])
        dubPath = dubins.path([0,0,0], finConf, self.rho, du.DubPathStr2Num(pathType))
        if dubPath is not None:
            du.PlotDubinsPath(dubPath,pathfmt)
            utils.PlotCircle2(self.tcircle, arcfmt) 
            utils.PlotArrow(finPt, finHdng, 1, arrowfmt) 
            plt.axis('equal')
            plt.show()
        return
    
    def P2CMinDubins(self):
        lengthsVec = []
        candPathsList = []
        
        ## Path local min LSL
        alpha, pathLength, seglengths = self.LocalMinLSL()
        if alpha is not None:
            cp = CandidateP2CPath('LSL',alpha, seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength)      

        ## Path local min RSL
        alpha, pathLength, seglengths = self.LocalMinRSL()
        if alpha is not None:
            cp = CandidateP2CPath('RSL',alpha, seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength)  

        if self.tcircle.circle_radius>2*self.rho:
            ## Path local min RSR
            alpha, pathLength, seglengths = self.LocalMinRSR()
            if alpha is not None:
                cp = CandidateP2CPath('RSR',alpha, seglengths)
                candPathsList.append(cp)
                lengthsVec.append(pathLength)  
            
            ## Path local min LSR
            alpha, pathLength, seglengths = self.LocalMinLSR()
            if alpha is not None:
                cp = CandidateP2CPath('LSR',alpha, seglengths)
                candPathsList.append(cp)
                lengthsVec.append(pathLength) 

        ## Path local min RLR
        alpha, pathLength, seglengths = self.LocalMinRLR()
        if alpha is not None:
            cp = CandidateP2CPath('RLR',alpha, seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength)  
            
        ## Path local min LRL
        alpha, pathLength, seglengths = self.LocalMinLRL()
        if alpha is not None:
            cp = CandidateP2CPath('LRL',alpha, seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength)                                           
        
        ## RLR Feasibility limits
        alpha, pathLength = P2CDub.RLRFeasLimits()
        if alpha is not None:
            cp = CandidateP2CPath('RLR',alpha, (0,0))
            candPathsList.append(cp)
            lengthsVec.append(pathLength)                                           
            
        ## LRL Feasibility limits
        alpha, pathLength = P2CDub.LRLFeasLimits()
        if alpha is not None:
            cp = CandidateP2CPath('LRL',alpha, (0,0))
            candPathsList.append(cp)
            lengthsVec.append(pathLength)                                           
        
        ## Path LS
        alpha, pathLength, seglengths = self.PathLS()
        if alpha is not None:
            cp = CandidateP2CPath('LS',alpha, seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength)  

        ## Path SL
        alpha, pathLength, seglengths = self.PathSL()
        if alpha is not None:
            cp = CandidateP2CPath('SL',alpha,seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength) 

        ## Path RS
        alpha, pathLength, seglengths = self.PathRS()
        if alpha is not None:
            cp = CandidateP2CPath('RS',alpha, seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength) 

        ## Path SR
        alpha, pathLength, seglengths = self.PathSR()
        if alpha is not None:
            cp = CandidateP2CPath('SR',alpha, seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength) 

        ## Path LR
        alpha, pathLength, seglengths = self.PathLR()
        if alpha is not None:
            cp = CandidateP2CPath('LR',alpha, seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength) 

        ## Path RL
        alpha, pathLength, seglengths = self.PathRL()
        if alpha is not None:
            cp = CandidateP2CPath('RL',alpha, seglengths)
            candPathsList.append(cp)
            lengthsVec.append(pathLength)    
        
        if lengthsVec:
            minInd = np.argmin(lengthsVec)                                        
        else:
            return None, None
        return candPathsList[minInd], candPathsList

    def PlotAllPaths(self, candPathsList):
        pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
        arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
        arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')
        for candPath in candPathsList:
            # print('candPath: ', candPath)                
            if candPath.pathType in ['LS', 'SL', 'RS', 'SR', 'LR', 'RL']:
                self.PlotDubPath(candPath.angPos, candPath.segLengths, candPath.pathType)
            else:
                self.PlotDubPath2(candPath.angPos, candPath.pathType)
                
        return
    
if __name__ == "__main__":

    LSL =0; LSR = 1; RSL = 2; RSR = 3; RLR = 4; LRL = 5; 
    iniConf = np.array([0,0,0])
    rho = 1    
    pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
    arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
    arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')
    
    P2CDub = P2CircleDubins((2.5, -1.5), 2.2, rho) #the arguments are circle center, circle radius and minimum turn radius
    # This class assumes start config is (0,0,0)        
    
    
    ######################## this script below could be used to test each individual path ################
    # alpha_star, length, segLengths = P2CDub.PathLS()
    # alpha_star, length, segLengths = P2CDub.LocalMinLSR()
    # alpha_star, pathLength = P2CDub.LRLFeasLimits()

    # print("path at alpha: ", alpha_star)
   
    # if alpha_star is not None:
    #     # P2CDub.PlotDubPath(alpha_star, segLengths, 'LSR')
    #     P2CDub.PlotDubPath2(alpha_star, 'LRL')

    ######################## example computes minmum Dubins path to circle ################

    
    
    minPath, candPathsList = P2CDub.P2CMinDubins()
    print('minPath: ', minPath)  
    # for pth in candPathsList: print(pth)

    if minPath:          
        P2CDub.PlotAllPaths([minPath])

    plt.show()
    
