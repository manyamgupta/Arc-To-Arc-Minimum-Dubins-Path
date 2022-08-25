from cmath import isfinite
import numpy as np
from numpy import pi,cos,sin
import matplotlib.pyplot as plt
import dubins
import dubutils as du 
import DubinsP2Arc as dubP2A
import utils
from types import SimpleNamespace
from dataclasses import dataclass
from timeit import default_timer as timer

# @dataclass
# class Arc:
#     cntr_x: float
#     cntr_y: float
#     arc_radius: float
#     # arc goes from lb to ub in ccw direction
#     angPos_lb: float # angular position lower bound
#     angPos_ub: float # angular position upper bound

@dataclass
class CandidatePath:
    pathType: str 
    angPos_arc1: float # angular position at first arc
    angPos_arc2: float # angular position at second arc
    segLengths: tuple
    
class Arc2ArcDubins:

    # def __init__(self, arc1_cntr, arc1_radius, arc1_bounds, arc2_cntr, arc2_radius, arc2_bounds, rho):     
    def __init__(self, arc1, arc2, rho):     
        
        # Assumption: Arc1 center is (0,0)
        
        # self.arc1 = Arc(arc1_cntr[0], arc1_cntr[1], arc1_radius, arc1_bounds[0], arc1_bounds[1])
        # self.arc2 = Arc(arc2_cntr[0], arc2_cntr[1], arc2_radius, arc2_bounds[0], arc2_bounds[1])
        self.arc1 = arc1
        self.arc2 = arc2        
        self.MoveArc1ToOrig() #This moves the arc1 and arc2 such that center of arc1 is at origin, and the
        self.rho = rho #minimum turn radius of the vehicle
        
        #length of the line connecting the centres of the two arcs
        self.len_c1c2 = np.linalg.norm(np.array([self.arc1.cntr_x-self.arc2.cntr_x, self.arc1.cntr_y-self.arc2.cntr_y]))
        
        #angle made the line c1c2 with x-axis (slope)
        self.psi_c1c2 = np.arctan2(self.arc2.cntr_y-self.arc1.cntr_y, self.arc2.cntr_x-self.arc1.cntr_x)
        
        
        arc1_ref = self.ReflectXaxis(self.arc1)
        arc2_ref = self.ReflectXaxis(self.arc2)
        
        self.arc2_ref_trans = utils.Arc(0,0, self.arc2.arc_radius, arc2_ref.angPos_lb, arc2_ref.angPos_ub)
        self.arc1_ref_trans = utils.Arc(arc1_ref.cntr_x-arc2_ref.cntr_x, arc1_ref.cntr_y-arc2_ref.cntr_y, self.arc1.arc_radius, arc1_ref.angPos_lb, arc1_ref.angPos_ub)
        
        # empty lists for lengths and candidate paths, these are populated when A2AMinDubins() is called
        self.lengthsVec = []
        self.candPathsList = []
        
        self.arc1_bound_config_lb = (self.arc1.cntr_x+self.arc1.arc_radius*np.cos(arc1.angPos_lb), self.arc1.cntr_y+self.arc1.arc_radius*np.sin(arc1.angPos_lb), arc1.angPos_lb-np.pi/2)
        self.arc1_bound_config_ub = (self.arc1.cntr_x+self.arc1.arc_radius*np.cos(arc1.angPos_ub), self.arc1.cntr_y+self.arc1.arc_radius*np.sin(arc1.angPos_ub), arc1.angPos_ub-np.pi/2)

        self.arc2_bound_config_lb = (self.arc2.cntr_x+self.arc2.arc_radius*np.cos(arc2.angPos_lb), self.arc2.cntr_y+self.arc2.arc_radius*np.sin(arc2.angPos_lb), arc2.angPos_lb-np.pi/2)
        self.arc2_bound_config_ub = (self.arc2.cntr_x+self.arc2.arc_radius*np.cos(arc2.angPos_ub), self.arc2.cntr_y+self.arc2.arc_radius*np.sin(arc2.angPos_ub), arc2.angPos_ub-np.pi/2)
        
        return
    
    def MoveArc1ToOrig(self):
        #This moves the arc1 and arc2 such that center of arc1 is at origin, and the
        self.arc2.cntr_x = self.arc2.cntr_x-self.arc1.cntr_x
        self.arc1.cntr_x = 0        
        self.arc2.cntr_y = self.arc2.cntr_y-self.arc1.cntr_y
        self.arc1.cntr_y = 0
        
        return
    
    def RotateTransArc(self, config, arc):
        # Rotates and translated the arc such that the config aligns with [0,0,0]
        cntr2_trans = (arc.cntr_x-config[0], arc.cntr_y-config[1])
        cntr2_trans_rot = utils.RotateVec2(cntr2_trans, -config[2])
        # arc_trans_rot = Arc(cntr2_trans_rot[0], cntr2_trans_rot[1], arc.arc_radius, arc.angPos_lb-config[2], arc.angPos_ub-config[2])
        
        return utils.Arc(cntr2_trans_rot[0], cntr2_trans_rot[1], arc.arc_radius, arc.angPos_lb-config[2], arc.angPos_ub-config[2])
    
    def ReflectXaxis(self, arc):
        # Reflection of an arc across x-axis
        arc_ref = utils.Arc(arc.cntr_x, -arc.cntr_y, arc.arc_radius, np.mod(-arc.angPos_ub, 2*np.pi), np.mod(-arc.angPos_lb, 2*np.pi))
        
        return arc_ref
    def LocalMinLS(self, arc1=None, arc2=None):
        # Local minimum of LS path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        if arc1 is None:
            arc1 = self.arc1        
            arc2 = self.arc2
            psi = self.psi_c1c2
        else:
            psi = np.arctan2(arc2.cntr_y-arc1.cntr_y, arc2.cntr_x-arc1.cntr_x)
                
        r1 = arc1.arc_radius
        r2 = arc2.arc_radius
        d = self.len_c1c2        
        rho = self.rho
        
        coeffs = [2*rho*d, d**2-r1**2+2*r2*rho-2*r1*rho, 2*r2*d, r2**2]
        roots = np.roots(coeffs)

        al1 = None
        al2 = None
        segLengths = (None, None)
        lengthLS = 100000000
        for eta in roots:   
            if np.imag(eta)==0 and np.abs(eta) <= 1:   
                
                eta_real = np.real(eta)
                al1_min = np.arccos((rho*eta_real**2 + d*eta_real + r2)/(r1*eta_real+rho*eta_real))+psi
                al2_min = np.arccos(eta_real)+psi
                phi = np.mod(al2_min-al1_min, 2*np.pi)
                len_S = d*np.sin(al1_min+phi-psi)-(r1+rho)*np.sin(phi)            
                # if utils.InInt(arc1.angPos_lb, arc1.angPos_ub, al1_min) and utils.InInt(arc2.angPos_lb, arc2.angPos_ub, al2_min) and len_S>=0:                
                if utils.CheckFeasibility(arc1, arc2, al1_min, rho, (rho*phi, len_S), 'LS') and len_S>=0:
                    if rho*phi+len_S < lengthLS:
                        al1 = al1_min
                        al2 = al2_min
                        segLengths = (rho*(phi), len_S)
                        lengthLS = rho*phi+len_S
                    
        return (al1, al2), segLengths
    def LocalMinSL(self):
        # Local minimum of SL path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        
        alphas, segLengths = self.LocalMinLS(self.arc2_ref_trans, self.arc1_ref_trans)
        
        if alphas[0]:
            alphsMinSL = [np.mod(-alphas[1], 2*np.pi),np.mod(-alphas[0], 2*np.pi)]
        else:
            return [None, None], [None, None]
        
        return alphsMinSL, segLengths[::-1]
    
    def LocalMinRS(self, arc1=None, arc2=None):
        # Local minimum of RS path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs

        if arc1 is None:
            arc1 = self.arc1        
            arc2 = self.arc2
            psi = self.psi_c1c2
        else:
            psi = np.arctan2(arc2.cntr_y-arc1.cntr_y, arc2.cntr_x-arc1.cntr_x)
                
        r1 = arc1.arc_radius
        r2 = arc2.arc_radius
        d = self.len_c1c2        
        rho = self.rho
        
        coeffs = [2*rho*d, -d**2+r1**2+2*r2*rho-2*r1*rho, -2*r2*d, -r2**2]

        roots = np.roots(coeffs)

        al1 = None
        al2 = None
        segLengths = (None, None)
        lengthLS = 100000000
        for eta in roots:   
            if np.imag(eta)==0 and np.abs(eta) <= 1:   
                
                eta_real = np.real(eta)
                al1_min = -np.arccos((-rho*eta_real**2 + d*eta_real + r2)/(r1*eta_real-rho*eta_real))+psi
                al2_min = np.arccos(eta_real)+psi
                phi = np.mod(al1_min-al2_min, 2*np.pi)
                len_S = d*np.sin(al1_min-phi-psi)+(r1-rho)*np.sin(phi)            
                # if utils.InInt(arc1.angPos_lb, arc1.angPos_ub, al1_min) and utils.InInt(arc2.angPos_lb, arc2.angPos_ub, al2_min) and len_S>=0:
                if utils.CheckFeasibility(arc1, arc2, al1_min, rho, (rho*phi, len_S), 'RS') and len_S>=0:
                
                    if rho*phi+len_S < lengthLS:
                        al1 = al1_min
                        al2 = al2_min
                        segLengths = (rho*(phi), len_S)
                        lengthLS = rho*phi+len_S
                    
        return (al1, al2), segLengths


    def LocalMinSR(self):
        # Local minimum of SR path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs

        alphas, segLengths = self.LocalMinRS(self.arc2_ref_trans, self.arc1_ref_trans)
        if alphas[0]:
            alphsMinSR = [np.mod(-alphas[1], 2*np.pi),np.mod(-alphas[0], 2*np.pi)]
        else:
            return [None, None], [None, None]
        
        return alphsMinSR, segLengths[::-1]

    def LocalMinLR(self, arc1=None, arc2=None):
        # Local minimum of LR path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs

        if arc1 is None:
            arc1 = self.arc1        
            arc2 = self.arc2
            psi = self.psi_c1c2
        else:
            psi = np.arctan2(arc2.cntr_y-arc1.cntr_y, arc2.cntr_x-arc1.cntr_x)
                
        r1 = arc1.arc_radius
        r2 = arc2.arc_radius
        d = self.len_c1c2        
        rho = self.rho
        
        c1 = 16*rho**3*d
        gamma = r1**2-r2**2+2*rho*(r1+r2)-d**2
        c2 = 12*rho**2*d**2 - 8*gamma*rho**2 - 16*rho**2*r2**2 +32*rho**3*r2
        c3 = -8*gamma*rho*d-16*rho*r2**2*d+32*rho**2*r2*d
        c4 = gamma**2 - 4*d**2*r2**2 + 8*rho*r2*d**2
        coeffs = [c1, c2, c3, c4]

        roots = np.roots(coeffs)

        al1 = None
        al2 = None
        segLengths = (None, None)
        lengthLR = 100000000
        for eta in roots:   
            if np.imag(eta)==0 and np.abs(eta) <= 1:   
                
                for pm in (+1, -1):
                    eta_real = np.real(eta)
                    al1_min = np.arcsin(pm*rho*np.sqrt(1-eta_real**2)/(r1+rho))+psi
                    phi1 = np.mod(pm*np.arccos(eta_real)+psi-al1_min, 2*np.pi)
                    
                    al2_min = np.pi+np.arcsin(rho*np.sqrt(1-eta_real**2)/(r2-rho))+psi
                    # al2_min = np.arccos(np.sqrt(r2**2-2*rho*r2+rho**2*eta**2)/(r2-rho))+psi 
                    phi2 = np.mod(al1_min-al2_min+phi1, 2*np.pi)
                    feas = utils.CheckFeasibility(arc1, arc2, al1_min, rho, (rho*phi1, rho*phi2), 'LR')
        
                    
                # if utils.InInt(arc1.angPos_lb, arc1.angPos_ub, al1_min) and utils.InInt(arc2.angPos_lb, arc2.angPos_ub, al2_min):
                    if feas and rho*(phi1+phi2) < lengthLR:
                        al1 = al1_min
                        al2 = al2_min
                        segLengths = (rho*phi1, rho*phi2)
                        lengthLR = rho*(phi1+phi2)
                    
        return (al1, al2), segLengths

    def LocalMinRL(self):
        # Local minimum of RL path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        
        alphas, segLengths = self.LocalMinLR(self.arc2_ref_trans, self.arc1_ref_trans)
        
        if alphas[0]:
            alphsMinSR = [np.mod(-alphas[1], 2*np.pi),np.mod(-alphas[0], 2*np.pi)]
        else:
            return [None, None], [None, None]
        
        return alphsMinSR, segLengths[::-1]

    def LocalMinLSL(self):
        # Local minimum of LSL path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs

        r1 = self.arc1.arc_radius
        r2 = self.arc2.arc_radius
        d = self.len_c1c2
        psi = self.psi_c1c2
        
        phi1 = np.mod(np.arccos(self.rho/(r1+self.rho)), 2*np.pi)
        phi2 = np.mod(np.arccos(self.rho/(r2+self.rho)), 2*np.pi)
        len_S = d - (r1+self.rho)*np.sin(phi1)-(r2+self.rho)*np.sin(phi2)
        if len_S <=0:
            return (None, None), (None, None, None)
        al1 = psi+np.pi/2-phi1
        al2 = al1+phi1+phi2
        if utils.InInt(self.arc1.angPos_lb, self.arc1.angPos_ub, al1) and utils.InInt(self.arc2.angPos_lb, self.arc2.angPos_ub, al2):
        
            return (al1, al2), (self.rho*phi1, len_S, self.rho*phi2)
        else:
            return (None, None), (None, None, None)
            
    def LocalMinRSR(self):
        # Local minimum of RSR path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        
        r1 = self.arc1.arc_radius
        r2 = self.arc2.arc_radius        
        d = self.len_c1c2
        psi = self.psi_c1c2
        if r1 < 2*self.rho or r2 < 2*self.rho:
            return (None, None), (None, None, None)
        
        phi1 = np.mod(np.pi+np.arccos(self.rho/(r1-self.rho)), 2*np.pi)
        phi2 = np.mod(np.pi+np.arccos(self.rho/(r2-self.rho)), 2*np.pi)

        len_S = d - (r1-self.rho)*np.sin(phi1-np.pi)-(r2-self.rho)*np.sin(phi2-np.pi)
        if len_S <=0:
            return (None, None), (None, None, None)
        al1 = psi-3*np.pi/2+phi1
        al2 = al1-phi1-phi2
        if utils.InInt(self.arc1.angPos_lb,self. arc1.angPos_ub, al1) and utils.InInt(self.arc2.angPos_lb, self.arc2.angPos_ub, al2):
        
            return (al1, al2), (self.rho*phi1, len_S, self.rho*phi2)
        else:
            return (None, None), (None, None, None)

    def LocalMinLSR(self, arc1=None, arc2=None):
        # Local minimum of LSR path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        if arc1 is None:
            arc1 = self.arc1        
            arc2 = self.arc2
            psi = self.psi_c1c2
        else:
            psi = np.arctan2(arc2.cntr_y-arc1.cntr_y, arc2.cntr_x-arc1.cntr_x)
            
        r1 = arc1.arc_radius
        r2 = arc2.arc_radius        
        d = self.len_c1c2
        rho = self.rho
        if r2 < 2*rho:
            return (None, None), (None, None, None)
        phi1 = np.arccos(rho/(r1+rho))
        phi2 = np.mod(np.pi+np.arccos(rho/(r2-rho)), 2*np.pi)
        len_S = d - (r1+rho)*np.sin(phi1)-(r2-rho)*np.sin(phi2-np.pi)
        if len_S <=0:
            return (None, None), (None, None, None)
        al1 = psi+np.pi/2-phi1
        al2 = al1+phi1-phi2
        if utils.InInt(self.arc1.angPos_lb, self.arc1.angPos_ub, al1) and utils.InInt(self.arc2.angPos_lb, self.arc2.angPos_ub, al2):    
            return (al1, al2), (rho*phi1, len_S, rho*phi2)
        else:
            return (None, None), (None, None, None)

    def LocalMinRSL(self):
        # Local minimum of RSR path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        
        alphas, segLengths = self.LocalMinLSR(self.arc2_ref_trans, self.arc1_ref_trans)
        # self.PlotDubPath(alphas, segLengths, 'LSR', self.arc2_ref_trans, self.arc1_ref_trans)
        # plt.show()
        if alphas[0]:
            alphsMinRSL = [np.mod(-alphas[1], 2*np.pi),np.mod(-alphas[0], 2*np.pi)]
        else:
            return (None, None), (None, None, None)
        
        return alphsMinRSL, segLengths[::-1]

    def LocalMinRLR(self):
        # Local minimum of RLR path from arc1 to arc2    
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        
        r1 = self.arc1.arc_radius
        r2 = self.arc2.arc_radius        
        d = self.len_c1c2
        psi = self.psi_c1c2
        rho = self.rho
        zeta = (r1-rho)**2+(r2-rho)**2
        c1 = 256*rho**4-64*rho**2
        c2 = -256*rho**3*d+32*rho*d
        c3 = 96*rho**2*d**2-32*rho**2*zeta+64*rho**4-4*d**2
        c4 = -16*rho*d**3+16*rho*d*zeta-32*rho**3*d
        c5 = d**4 + zeta**2 - 2*zeta*d**2 + 4*rho**2*d**2 -4*(r1-rho)**2*(r2-rho)**2
        
        coeffs = [c1, c2, c3, c4, c5]
        roots = np.roots(coeffs)
        lengthRLR = 10000000
        al1, al2 = None, None 
        segLengths = (None, None, None)
        for lamda in roots:   
            if np.imag(lamda)==0 and lamda > 0:   
                
                lamda_real = np.real(lamda)
                if rho**2- lamda_real**2>0:
                    # al1 = np.pi - np.arcsin(np.sqrt(rho**2- lamda_real**2)/(r1-rho))
                    beta = 2*np.arcsin(lamda_real/rho)
                    gamma = np.arcsin(np.sqrt(rho**2-lamda_real**2)/(r1-rho))
                    al1_min = np.pi-gamma+psi
                    phi1 = np.mod(3*np.pi/2 -gamma - beta/2, 2*np.pi)
                    phi2 = np.mod(2*np.pi-beta, 2*np.pi)
                    phi3 = 3*np.pi/2 - beta/2 - np.arcsin(np.sqrt(rho**2-lamda_real**2)/(r2-rho))
                    phi3 = np.mod(phi3, 2*np.pi)
                    al2_min = al1_min-phi1+phi2-phi3
                    
                    # al1 = al1_min
                    # al2 = al2_min
                    # feas = utils.InInt(self.arc1.angPos_lb, self.arc1.angPos_ub, al1_min) and utils.InInt(self.arc2.angPos_lb, self.arc2.angPos_ub, al2_min)
                    feas = utils.CheckFeasibility(self.arc1, self.arc2, al1_min, rho, (rho*phi1, rho*phi2, rho*phi3), 'RLR')
                
                    if feas and rho*(phi1+phi2+phi3) < lengthRLR:
                        al1 = al1_min
                        al2 = al2_min
                        # segLengths = [rho*phi1, rho*phi2, rho*phi3]
                        lengthRLR = rho*(phi1+phi2+phi3)
                        segLengths = (rho*phi1, rho*phi2, rho*phi3)
                        
        return (al1, al2), segLengths
        
    def PathR_A2A(self):
        # path with single right arc from arc1 to arc2
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        
        if self.len_c1c2 > self.arc1.arc_radius+self.arc2.arc_radius+2*self.rho - 2*self.rho or self.arc1.arc_radius>self.len_c1c2+self.arc2.arc_radius or self.arc2.arc_radius>self.len_c1c2+self.arc1.arc_radius:
            return (None, None), None
        al1a = self.psi_c1c2+ np.arccos( ((self.arc1.arc_radius - self.rho)**2 + self.len_c1c2**2 - ((self.arc2.arc_radius - self.rho)**2 ))/(2*self.len_c1c2*(self.arc1.arc_radius - self.rho)))        
        phia = 2*np.pi - np.arccos( ((self.arc1.arc_radius - self.rho)**2 - self.len_c1c2**2 + ((self.arc2.arc_radius - self.rho)**2 ))/(2*(self.arc2.arc_radius - self.rho)*(self.arc1.arc_radius - self.rho)))
        phia = np.mod(phia,2*np.pi)
        
        al1b = self.psi_c1c2 -np.arccos( ((self.arc1.arc_radius - self.rho)**2 + self.len_c1c2**2 - ((self.arc2.arc_radius - self.rho)**2 ))/(2*self.len_c1c2*(self.arc1.arc_radius - self.rho)))
        phib = np.arccos( ((self.arc1.arc_radius - self.rho)**2 - self.len_c1c2**2 + ((self.arc2.arc_radius - self.rho)**2 ))/(2*(self.arc2.arc_radius - self.rho)*(self.arc1.arc_radius - self.rho)))
        phib = np.mod(phib,2*np.pi)
        phiVec = (phia, phib)
        arclength = 10000000.
        al1Min, al2Min = None, None
        for indx, al1 in enumerate([al1a, al1b]):
            phi = phiVec[indx]
            al2 = np.mod(al1-phi, 2*np.pi)
            if utils.InInt(self.arc1.angPos_lb, self.arc1.angPos_ub, al1) and utils.InInt(self.arc2.angPos_lb, self.arc2.angPos_ub, al2):
                if self.rho*phi< arclength:
                    al1Min = al1
                    al2Min = al2
                    arclength = self.rho*phi
                    
        return (al1Min, al2Min), arclength
            
    def PathL_A2A(self):
        # path with single Left arc from arc1 to arc2
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        
        if self.len_c1c2 > self.arc1.arc_radius+self.arc2.arc_radius+2*self.rho:
            return (None, None), None
        al1 = self.psi_c1c2+ np.arccos( ((self.arc1.arc_radius+self.rho)**2 + self.len_c1c2**2 - ((self.arc2.arc_radius+self.rho)**2 ))/(2*self.len_c1c2*(self.arc1.arc_radius+self.rho)))
        
        phi = np.arccos( ((self.arc1.arc_radius+self.rho)**2 - self.len_c1c2**2 + ((self.arc2.arc_radius+self.rho)**2 ))/(2*(self.arc2.arc_radius+self.rho)*(self.arc1.arc_radius+self.rho)))
        phi = np.mod(phi,2*np.pi)        
        al2 = al1+phi
        
        al1b = self.psi_c1c2 - np.arccos( ((self.arc1.arc_radius+self.rho)**2 + self.len_c1c2**2 - ((self.arc2.arc_radius+self.rho)**2 ))/(2*self.len_c1c2*(self.arc1.arc_radius+self.rho)))
        phib = -np.arccos( ((self.arc1.arc_radius+self.rho)**2 - self.len_c1c2**2 + ((self.arc2.arc_radius+self.rho)**2 ))/(2*(self.arc2.arc_radius+self.rho)*(self.arc1.arc_radius+self.rho)))
        phib = np.mod(phib, 2*np.pi)        
        al2b = al1b+phib
        
        if utils.InInt(self.arc1.angPos_lb, self.arc1.angPos_ub, al1) and utils.InInt(self.arc2.angPos_lb, self.arc2.angPos_ub, al2):
            return (al1, al2), self.rho*phi   
        elif utils.InInt(self.arc1.angPos_lb, self.arc1.angPos_ub, al1b) and utils.InInt(self.arc2.angPos_lb, self.arc2.angPos_ub, al2b):
            return (al1b, al2b), self.rho*phib   
             
        else:
            return (None, None), None
    
    def PathS_A2A(self):
        # path with one straight line
        # Assumption: center of arc1 is [0,0], tangents are clockwise on arcs
        
        al1 = np.pi/2+self.psi_c1c2+ np.arcsin( (self.arc2.arc_radius- self.arc1.arc_radius)/self.len_c1c2 )
        len_S = np.sqrt(self.len_c1c2**2 -(self.arc2.arc_radius- self.arc1.arc_radius)**2 )
        if utils.InInt(self.arc1.angPos_lb, self.arc1.angPos_ub, al1) and utils.InInt(self.arc2.angPos_lb, self.arc2.angPos_ub, al1):
            return (al1, al1), len_S
        else:
            return (None, None), None
            
    def PlotA2APath(self, alphas, segLengths, pathType, arc1=None, arc2=None, pathfmt=None):
        # Plots a dubins path from arc1 to arc2
        # alphas are the start and end positions on the arcs are
        # seglengths are the length of each segment of the Dubins path
        # pathType is the dubins mode
        # is arc1 and arc2 are not given, the arcs from the main Arc2ArcDubins object are used
        if arc1 is None:
            arc1 = self.arc1
            arc2 = self.arc2
        if pathfmt is None:
            pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
        if alphas[0]:            
            arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
            arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')
            al1 = alphas[0]
            al2 = alphas[1]    
            iniPt = np.array([arc1.cntr_x+arc1.arc_radius*np.cos(al1), arc1.cntr_y+arc1.arc_radius*np.sin(al1)])
            iniHdng = al1-np.pi/2
            iniConf_min = np.array([iniPt[0], iniPt[1], iniHdng])           
            finPt = np.array([arc2.cntr_x+arc2.arc_radius*np.cos(al2), arc2.cntr_y+arc2.arc_radius*np.sin(al2)])
            finHdng = al2-np.pi/2     
            du.PlotDubPathSegments(iniConf_min, pathType, segLengths, self.rho, pathfmt)
            
            utils.PlotArc(arc1, arcfmt)
            utils.PlotArc(arc2, arcfmt)        
            utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
            utils.PlotArrow(finPt, finHdng, 1, arrowfmt)  
            utils.PlotLineSeg([arc1.cntr_x, arc1.cntr_y], [arc2.cntr_x, arc2.cntr_y], arrowfmt)      
            plt.axis('equal')
            # plt.show()
        return
    
    def PlotAllPaths(self, candPathsList):
        # Plots the list of paths from candPathsList
        # Prints the parameters for each path
        
        pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
        arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
        arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')
        for candPath in candPathsList:
            print('candPath: ', candPath)    
            al1 = candPath.angPos_arc1
            
            iniPt = np.array([self.arc1.cntr_x+self.arc1.arc_radius*np.cos(al1), self.arc1.cntr_y+self.arc1.arc_radius*np.sin(al1)])
            iniHdng = al1-np.pi/2
            iniConf_min = np.array([iniPt[0], iniPt[1], iniHdng])  
            plt.figure()                         
            du.PlotDubPathSegments(iniConf_min, candPath.pathType, candPath.segLengths, self.rho, pathfmt)
            
            utils.PlotArc(self.arc1, arcfmt)
            utils.PlotArc(self.arc2, arcfmt)   
            al2 = candPath.angPos_arc2
            finPt = np.array([self.arc2.cntr_x+self.arc2.arc_radius*np.cos(al2), self.arc2.cntr_y+self.arc2.arc_radius*np.sin(al2)])
            finHdng = al2-np.pi/2      
            utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
            utils.PlotArrow(finPt, finHdng, 1, arrowfmt)  
            utils.PlotLineSeg([self.arc1.cntr_x, self.arc1.cntr_y], [self.arc2.cntr_x, self.arc2.cntr_y], arrowfmt)      
            plt.axis('equal')
        # plt.show()
        return
    def A2AMinDubins(self):
        # Computes the list of candidate paths for minimum arc to arc dubins paths        
        # returns the length of the minimum path, minimum path, and candidate paths
        
        ######################## One Segment Paths ########################
        # Path S
        alphas, segLength = self.PathS_A2A()
        if alphas[0]:
            cp = CandidatePath('S',alphas[0], alphas[1], (segLength,0.))
            self.candPathsList.append(cp)
            self.lengthsVec.append(segLength)
        
        ## Path L
        alphas, segLength = self.PathL_A2A()
        if alphas[0]:
            cp = CandidatePath('L',alphas[0], alphas[1], (segLength,0.))
            self.candPathsList.append(cp)
            self.lengthsVec.append(segLength)
        
        ## Path R
        alphas, segLength = self.PathR_A2A()
        if alphas[0]:
            cp = CandidatePath('R',alphas[0], alphas[1], (segLength,0.))
            self.candPathsList.append(cp)
            self.lengthsVec.append(segLength) 
            
        
        ######################## Local minima of the paths with one degree of freedom ########################
        
        ## Path LS
        alphas, segLengths = self.LocalMinLS()
        if alphas[0]:
            cp = CandidatePath('LS',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
            
        ## Path SL
        alphas, segLengths = self.LocalMinSL()
        if alphas[0]:
            cp = CandidatePath('SL',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))

        ## Path RS
        alphas, segLengths = self.LocalMinRS()
        if alphas[0]:
            cp = CandidatePath('RS',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
        ## Path SR
        alphas, segLengths = self.LocalMinSR()
        if alphas[0]:
            cp = CandidatePath('SR',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
        ## Path LR
        alphas, segLengths = self.LocalMinLR()
        if alphas[0]:
            cp = CandidatePath('LR',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
        ## Path RL
        alphas, segLengths = self.LocalMinRL()
        if alphas[0]:
            cp = CandidatePath('RL',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
        ## Path LSL
        alphas, segLengths = self.LocalMinLSL()
        if alphas[0]:
            cp = CandidatePath('LSL',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
        ## Path RSR
        alphas, segLengths = self.LocalMinRSR()
        if alphas[0]:
            cp = CandidatePath('RSR',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
        ## Path LSR
        alphas, segLengths = self.LocalMinLSR()
        if alphas[0]:
            cp = CandidatePath('LSR',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
        ## Path RSL
        alphas, segLengths = self.LocalMinRSL()
        if alphas[0]:
            cp = CandidatePath('RSL',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
        ## Path RLR
        alphas, segLengths = self.LocalMinRLR()
        if alphas[0]:
            cp = CandidatePath('RLR',alphas[0], alphas[1], segLengths)
            self.candPathsList.append(cp)
            self.lengthsVec.append(sum(segLengths))
        
        ######################## Boudnary candidate ########################            
        ## candidate paths where one end of the path is at the boudnary of an arc
        ## other end could be anywhere on the second arc
        self.OneBoundaryPaths()

        ######################## Boudnary to boundary candidates ########################            
        ## candidate paths where both ends of the path are at the boudnaries of the arcs
        self.TwoBoundaryPaths()

        if self.lengthsVec:
            minInd = np.argmin(self.lengthsVec)
        else:
            return None, None, None
        return self.lengthsVec[minInd], self.candPathsList[minInd], self.candPathsList
    
    def TwoBoundaryPaths(self):
        # This computed the four dubins paths from the boudnary limits of the first arc to boundary limits of the second arc
        for iniConf in [self.arc1_bound_config_lb, self.arc1_bound_config_ub]:
            for finConf in [self.arc2_bound_config_lb, self.arc2_bound_config_ub]:
                path_bnd2bnd = dubins.shortest_path(iniConf, finConf, self.rho)
                # pathType = du.DubPathTypeNum2Str(path_bnd2bnd.path_type())
                cp = CandidatePath(du.DubPathTypeNum2Str(path_bnd2bnd.path_type()), iniConf[2]+np.pi/2, finConf[2]+np.pi/2, (path_bnd2bnd.segment_length(0), path_bnd2bnd.segment_length(1), path_bnd2bnd.segment_length(2)))
                self.candPathsList.append(cp)
                self.lengthsVec.append(path_bnd2bnd.path_length())                
        return
    
    def OneBoundaryPaths(self):

        ## lower bound and upper bound of arc1 to any config on arc2: shortest path
        
        for angPos_bnd in [self.arc1.angPos_lb, self.arc1.angPos_ub]:            
            arc1_bound_config = (self.arc1.cntr_x+self.arc1.arc_radius*np.cos(angPos_bnd), self.arc1.cntr_y+self.arc1.arc_radius*np.sin(angPos_bnd), angPos_bnd-np.pi/2)
            arc2_trans = self.RotateTransArc(arc1_bound_config, self.arc2)
            p2a_arc1bnd2arc2 = dubP2A.P2ArcDubins((arc2_trans.cntr_x, arc2_trans.cntr_y), arc2_trans.arc_radius, (arc2_trans.angPos_lb, arc2_trans.angPos_ub),  self.rho)
            minPath_p2a, _ = p2a_arc1bnd2arc2.P2AMinDubins()
            if minPath_p2a:
                cp1 = CandidatePath(minPath_p2a.pathType, angPos_bnd, minPath_p2a.angPos+angPos_bnd-np.pi/2, minPath_p2a.segLengths)
                self.candPathsList.append(cp1)
                self.lengthsVec.append(sum(minPath_p2a.segLengths))
        
        ## shortest path from any point on the first arc to the boundary limits of the second arc
        ## this is done by translating and reflecting, and uses point to arc function P2AMinDubins()
        for angPos_bnd in [self.arc2.angPos_lb, self.arc2.angPos_ub]:
            arc2_bound_config = (self.arc2.cntr_x+self.arc2.arc_radius*np.cos(angPos_bnd), self.arc2.cntr_y+self.arc2.arc_radius*np.sin(angPos_bnd), angPos_bnd+np.pi/2)
            arc1_trans = self.RotateTransArc(arc2_bound_config, self.arc1)
            arc1_trans_ref = self.ReflectXaxis(arc1_trans)
            p2a_arc2bnd2arc1 = dubP2A.P2ArcDubins((arc1_trans_ref.cntr_x, arc1_trans_ref.cntr_y), arc1_trans_ref.arc_radius, (arc1_trans_ref.angPos_lb, arc1_trans_ref.angPos_ub),  self.rho)
            minPath_p2a, _ = p2a_arc2bnd2arc1.P2AMinDubins()  
            if minPath_p2a:
                cp1 = CandidatePath(minPath_p2a.pathType[::-1], -minPath_p2a.angPos+angPos_bnd+np.pi/2, angPos_bnd, minPath_p2a.segLengths[::-1])
                self.candPathsList.append(cp1)
                self.lengthsVec.append(sum(minPath_p2a.segLengths))
        
        
        return
    
if __name__ == "__main__":

    LSL =0; LSR = 1; RSL = 2; RSR = 3; RLR = 4; LRL = 5;     
    rho = 1   
    tic = timer()
    # A2ADub = Arc2ArcDubins([0, 0], 2.6, [2.5, 6.28], [-4.5, -5], 2.8, [.5, 4.28], 1) 

    # A2ADub = Arc2ArcDubins([0, 0], 2.5, [0.01, 6.28], [1.5, 4], 2.8, [0.01, 6.28], 1) # for RL
    # A2ADub = Arc2ArcDubins([0, 0], 2.8, [1.51, 6.2], [-1.5, 3.75], 2.5, [0.01, 6.2], 1) #for LR
    
    # A2ADub = Arc2ArcDubins([0, 0], 2.5, [0.01, 6.28], [.6, 0.4], 2.7, [0.01, 6.28], 1) # for RLR
    
    arc1 = utils.Arc(0, 0, 2.6, 0.01, 3.28)
    arc2 = utils.Arc(5., 4., 2.8, 3.4, 6.28)

    A2ADub = Arc2ArcDubins(arc1, arc2, 1) 

    # alphas, segLengths = A2ADub.LocalMinRLR()
    alphas, segLengths = A2ADub.PathL_A2A()
    print('alphas: ', alphas)    
    print('segLengths: ', segLengths)
    if alphas[0]:
        A2ADub.PlotA2APath(alphas, (segLengths,0), 'L')        
    plt.show()
    # A2ADub.OneBoundaryPaths()

    
    # minLength, minPath, candPathsList = A2ADub.A2AMinDubins()   
    # comp_time = timer()-tic
    
    # print('minLength: ', minLength)  
    # print('comp_time: ', comp_time)  
    # if minPath:    
    #     A2ADub.PlotAllPaths(candPathsList)
    #     print('minPath: ', minPath)  
    #     A2ADub.PlotA2APath((minPath.angPos_arc1, minPath.angPos_arc2), minPath.segLengths, minPath.pathType)
    
    
    
    ############################# Test local min LS #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.1, 2.2)
    # arc2 = utils.Arc(7,2.5, 2.5, 2.1, 4.2)
    # alphas, segLengths = LocalMinLS(arc1, arc2, rho)
    # print('segLengths: ', segLengths)
    
    # if np.isfinite(alphas[0]):
    #     al1 = alphas[0]
    #     al2 = alphas[1]    
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minLS = np.array([iniPt[0], iniPt[1], iniHdng])           
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minLS, 'LS', segLengths, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)  
    #     utils.PlotLineSeg([arc1.c_x, arc1.c_y], [arc2.c_x, arc2.c_y], arrowfmt)      
    #     plt.axis('equal')
    #     plt.show()
    
    ############################# Test local min RS #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.1, 6.2)
    # arc2 = utils.Arc(7,2.5, 2.5, 0.1, 6.2)
    # alphas, segLengths = LocalMinRS(arc1, arc2, rho)
    # print('segLengths: ', segLengths)
    
    # if np.isfinite(alphas[0]):
    #     al1 = alphas[0]
    #     al2 = alphas[1]    
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minLS = np.array([iniPt[0], iniPt[1], iniHdng])           
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minLS, 'RS', segLengths, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt) 
    #     utils.PlotLineSeg([arc1.c_x, arc1.c_y], [arc2.c_x, arc2.c_y], arrowfmt)       
    #     plt.axis('equal')
    #     plt.show()
    
############################# Test local min SL #############################
    

    # arc1 = utils.Arc(0,0, 2.5, 0.1, 6.2)
    # arc2 = utils.Arc(-6, 4, 2.5, 0.1, 6.2)
    # alphas, segLengths = LocalMinSL(arc1, arc2, rho)
    
    # print('alphas: ', alphas)    
    # print('segLengths: ', segLengths)
    
    # if np.isfinite(alphas[0]):
    #     al1 = alphas[0]
    #     al2 = alphas[1]    
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minLS = np.array([iniPt[0], iniPt[1], iniHdng])           
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minLS, 'SL', segLengths, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)  
    #     utils.PlotLineSeg([arc1.c_x, arc1.c_y], [arc2.c_x, arc2.c_y], arrowfmt)      
    #     plt.axis('equal')
    #     plt.show()

############################# Test local min SR #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.1, 6.2)
    # arc2 = utils.Arc(6, 4, 2.5, 0.1, 6.2)
    # alphas, segLengths = LocalMinSR(arc1, arc2, rho)
    
    # print('alphas: ', alphas)    
    # print('segLengths: ', segLengths)
    
    # if np.isfinite(alphas[0]):
    #     al1 = alphas[0]
    #     al2 = alphas[1]    
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minLS = np.array([iniPt[0], iniPt[1], iniHdng])           
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minLS, 'SR', segLengths, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)  
    #     utils.PlotLineSeg([arc1.c_x, arc1.c_y], [arc2.c_x, arc2.c_y], arrowfmt)      
    #     plt.axis('equal')
    #     plt.show()
        
############################# Test local min LR #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.01, 6.28)
    # arc2 = utils.Arc(-3.5, 3, 2.5, 0.01, 6.28)
    # alphas, segLengths = LocalMinLR(arc1, arc2, rho)
    
    # print('alphas: ', alphas)    
    # print('segLengths: ', segLengths)
    
    # if np.isfinite(alphas[0]):
    #     al1 = alphas[0]
    #     al2 = alphas[1]    
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minLS = np.array([iniPt[0], iniPt[1], iniHdng])           
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minLS, 'LR', segLengths, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)  
    #     utils.PlotLineSeg([arc1.c_x, arc1.c_y], [arc2.c_x, arc2.c_y], arrowfmt)      
    #     plt.axis('equal')
    #     plt.show()
    
############################# Test local min RL #############################
    
    # arc1 = utils.Arc(0,0, 2.5, 0.01, 6.28)
    # arc2 = utils.Arc(-3.5, -3, 2.5, 0.01, 6.28)
    # alphas, segLengths = LocalMinRL(arc1, arc2, rho)
    
    # print('alphas: ', alphas)    
    # print('segLengths: ', segLengths)
    
    # if np.isfinite(alphas[0]):
    #     al1 = alphas[0]
    #     al2 = alphas[1]    
    #     iniPt = np.array([arc1.c_x+arc1.arc_radius*np.cos(al1), arc1.c_y+arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minLS = np.array([iniPt[0], iniPt[1], iniHdng])           
    #     finPt = np.array([arc2.c_x+arc2.arc_radius*np.cos(al2), arc2.c_y+arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minLS, 'RL', segLengths, rho, pathfmt)
        
    #     utils.PlotArc(arc1, arcfmt)
    #     utils.PlotArc(arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)  
    #     utils.PlotLineSeg([arc1.c_x, arc1.c_y], [arc2.c_x, arc2.c_y], arrowfmt)      
    #     plt.axis('equal')
    #     plt.show()
    
############################# Test local min CSC #############################
    
    # arc1 = utils.Arc(0, 0, 2.5, 0.01, 6.28)
    # arc2 = utils.Arc(-2, 4, 3, 0.01, 6.28)
    
    # # arc1 = utils.Arc(0, 0, 2.5, 0.01, 6.28)
    # # arc2 = utils.Arc(-.6, 0.4, 2.7, 0.01, 6.28)
    
    # alphas, segLengths = LocalMinRSR(arc1, arc2, rho)

    
    
    # if np.isfinite(alphas[0]):
    #     al1 = alphas[0]
    #     al2 = alphas[1]    
    #     iniPt = np.array([A2ADub.arc1.cntr_x+A2ADub.arc1.arc_radius*np.cos(al1), A2ADub.arc1.cntr_y+A2ADub.arc1.arc_radius*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
    #     iniConf_minLS = np.array([iniPt[0], iniPt[1], iniHdng])           
    #     finPt = np.array([A2ADub.arc2.cntr_x+A2ADub.arc2.arc_radius*np.cos(al2), A2ADub.arc2.cntr_y+A2ADub.arc2.arc_radius*np.sin(al2)])
    #     finHdng = al2-np.pi/2     
    #     du.PlotDubPathSegments(iniConf_minLS, 'LSL', segLengths, rho, pathfmt)
        
    #     utils.PlotArc(A2ADub.arc1, arcfmt)
    #     utils.PlotArc(A2ADub.arc2, arcfmt)        
    #     utils.PlotArrow(iniPt, iniHdng, 1, arrowfmt)
    #     utils.PlotArrow(finPt, finHdng, 1, arrowfmt)  
    #     utils.PlotLineSeg([A2ADub.arc1.cntr_x, A2ADub.arc1.cntr_y], [A2ADub.arc2.cntr_x, A2ADub.arc2.cntr_y], arrowfmt)      
    #     plt.axis('equal')
    #     plt.show()