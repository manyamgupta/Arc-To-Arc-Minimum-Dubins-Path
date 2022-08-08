import numpy as np
from numpy import pi,cos,sin
import matplotlib.pyplot as plt
# import dubins
import dubutils as du 
import utils
from types import SimpleNamespace

def PathLS(arc1, arc2, al1, rho):
    # Assumption: cenetr of the first arc (0,0)
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc1.arc_radius
    r2 = arc2.arc_radius
    lx = c_x-(r1+rho)*np.cos(al1)
    ly = c_y-(r1+rho)*np.sin(al1)
    
    
    dist_o1c2 = np.sqrt(lx**2 + ly**2)
    if np.abs((rho+r2)/(dist_o1c2)) > 1:
        return np.nan, [np.nan, np.nan, np.nan, np.nan, np.nan ], []
    psi1 = np.arctan2(ly, lx)
    psi2 = np.arcsin((rho+r2)/dist_o1c2)
    theta1 = al1-np.pi/2
    
    phi1 = np.mod(psi1+psi2-theta1, 2*pi)
    
    Ls = np.sqrt(lx**2 + ly**2 - (rho+r2)**2 )
    lengthLS = Ls + rho*phi1
    
    inf_x = (r1+rho)*np.cos(al1) + rho*cos(psi1+psi2-np.pi/2)
    inf_y = (r1+rho)*np.sin(al1) + rho*sin(psi1+psi2-np.pi/2)
    
    return lengthLS, [rho*phi1, Ls], [inf_x, inf_y]

def CheckLSMin(arc1, arc2, al, rho):
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc1.arc_radius
    r2 = arc2.arc_radius
    lx = c_x-(r1+rho)*np.cos(al)
    ly = c_y-(r1+rho)*np.sin(al)

    O1 = (r1+rho)*np.array([np.cos(al), np.sin(al)])

    iniPos = O1*r1/(r1+rho)
    iniConf = [iniPos[0], iniPos[1], al-np.pi/2]
    lengthLS, segLengths, infPt =  PathLS(arc1, arc2, al, rho)
    infPt = np.array(infPt)

    # O1P = infPt-O1
    C2 = np.array([c_x, c_y])
    # print('norm of O1P: ', np.linalg.norm(O1P))
    # doc_x = np.linalg.norm(infPt)
    # exp2 = np.linalg.norm(O1-C2)**2-(np.linalg.norm(C2)-doc_x)**2-(r2+rho)**2+r2**2
    # print('exp2: ', exp2)

    phi = segLengths[0]/rho
    lamda = segLengths[1]

    d = np.linalg.norm(C2)
    psi = np.arctan2(c_y, c_x)

    exp0 = lamda + (r1+rho)*np.sin(phi)-d*np.sin(al+phi-psi)
    exp1 = np.sqrt((r1+rho)**2 + rho**2-2*(r1+rho)*rho*np.cos(phi) ) -d - (r2/np.cos(al+phi-psi))
    # exp1 = np.sqrt((r1+rho)**2 + rho**2-2*(r1+rho)*rho*np.cos(phi) ) + np.sqrt(r2**2+lamda**2) - d

    exp2 = -r2*np.tan(al+phi-psi)+(r1+rho)*np.sin(phi)-d*np.sin(al+phi-psi)

    eta = np.cos(al+phi-psi)

    sg = np.sin(al+phi-psi)
    cg = np.cos(al+phi-psi)
    tg = np.tan(al+phi-psi)

    # exp3 = 4*(rho**2)*((r1+rho)**2) - 4*(rho**2)*(d*sg+r2*tg)**2-((d+r2/eta)**2 - (r1+rho)**2-rho**2 )**2
    # exp3 = 4*(rho**2)*((r1+rho)**2) - 4*(rho**2)*(d*sg+r2*tg)**2-((d+r2/eta)**2 - (r1+rho)**2-rho**2 )**2

    # exp3 = eta**4*(2*(rho**2)*(r1+rho)**2 - (r1+rho)**4 - rho**4) - 4*(rho**2)*(eta**2)*( d**2*eta**2 - d**2*eta**4+ r2**2 - r2**2*eta**2 + 2*r2*d*eta - 2*r2*d*eta**3 ) - (r2+d*eta)**4 + 2*eta**2*(r2+d*eta)**2*((r1+rho)**2+rho**2)

    # exp3 = (eta**4)*(2*rho**2*(r1+rho)**2 - (r1+rho)**4 - rho**4 - 4*(rho**2)*(d**2) +4*(rho**2)*(r2**2)-d**4 + 2*(d**2)*( (r1+rho)**2 + rho**2) )
    # exp3 += (eta**6)*4*(rho**2)*(d**2) + 8*(rho**2)*r2*d*(eta**5)
    # exp3 += (eta**3)*( -4*r2*d**3 + 4*r2*d*(r1**2+2*r1*rho) )
    # exp3 += (eta**2)*(-6*(r2**2)*(d**2)+2*(r2**2)*((r1**2)+2*r1*rho) )
    # exp3 += eta*(-4*(r2**3)*d) - r2**4

    # exp4 = (r1+rho)**2-(rho**2)*(sg**2)-(d+r2/cg+rho*cg)**2
    exp4 = 2*rho*d*eta**3 + (eta**2)*(d**2-r1**2+2*r2*rho-2*r1*rho) + 2*r2*d*eta + r2**2

    coeffs = [2*rho*d, d**2-r1**2+2*r2*rho-2*r1*rho, 2*r2*d, r2**2]

    roots = np.roots(coeffs)

    eta = roots[1]
    minAl = np.arccos((rho*eta**2 + d*eta + r2)/(r1*eta+rho*eta))+psi
    print('minAl1: ', minAl)

    eta = roots[2]
    minAl = np.arccos((rho*eta**2 + d*eta + r2)/(r1*eta+rho*eta))+psi
    print('minAl2: ', minAl)

    # print('roots: ', roots)

    # print('exp0: ', exp0)
    # print('exp1: ', exp1)
    # print('exp2: ', exp2)
    # print('exp3: ', exp3)
    # print('exp4: ', exp4)

    # exp5 = (r1+rho)*d*np.sin(al-psi) - lamda*rho - (r1+rho)*rho*np.sin(phi)
    # exp5 = np.sin(al+phi-psi)-(r1+rho)*np.sin(al-psi)/rho
    # print('exp5: ', exp5)

    exp6 = -((r1+rho)*rho/(lx**2+ly**2)/lamda)*(lamda*d*np.cos(al-psi)-(r1+rho)*lamda+(r2+rho)*d*np.sin(al-psi))-rho
    exp6 += (r1+rho)*d*np.sin(al-psi)/lamda
    print('exp6: ', exp6) 


    

    return


def LenLSPrime(arc1, arc2, rho, al):

    r1 = arc1.arc_radius
    r2 = arc2.arc_radius
    lx = arc2.c_x-(r1+rho)*np.cos(al)
    ly = arc2.c_y-(r1+rho)*np.sin(al)
    doc = np.sqrt(lx**2 + ly**2)
    Ls = np.sqrt(doc**2-(rho+r2)**2)
    psi1 = np.arctan2(ly, lx)
    psi2 = np.arcsin((rho+r2)/doc)

    # Ls_p = (r1+rho)*(lx*np.sin(al)-ly*np.cos(al))/Ls
    
    Ls_prime = (r1+rho)*np.sin(al-psi1)/(np.cos(psi2))
    psi1_p = -(r1+rho)*np.cos(al-psi1)/doc
    psi2_p = -(r1+rho)*(r2+rho)*np.sin(al-psi1)/(np.cos(psi2)*doc*doc)
    phi1_p = psi1_p+psi2_p-1
    
    # phi1_prime = ((r1+rho)/(doc*np.cos(psi2)))*np.cos(minAl-psi1+psi2)-1
    
    lenLS_prime = Ls_prime + rho*phi1_p
    # lenLS_prime = (r1+rho)*(np.sin(al-psi1) - (rho/doc)*np.cos(al-psi1-psi2))/(np.cos(psi2))-rho

    return lenLS_prime

def LenLSPrime2(arc1, arc2, rho, al):
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc1.arc_radius
    r2 = arc2.arc_radius
    lx = arc2.c_x-(r1+rho)*np.cos(al)
    ly = arc2.c_y-(r1+rho)*np.sin(al)
    doc = np.sqrt(lx**2 + ly**2)
    Ls = np.sqrt(doc**2-(rho+r2)**2)
    
    if np.abs((rho+r2)/(doc)) > 1:
        return np.nan
    psi1 = np.arctan2(ly, lx)
    psi2 = np.arcsin((rho+r2)/doc)
    d = np.sqrt(c_x**2+c_y**2)
    psi = np.arctan2(c_y, c_x)
    phi = np.mod(psi1+psi2-al1+np.pi/2, 2*pi)

    lenLS_prime = (r1+rho)*(d*np.sin(al-psi)-rho*np.sin(phi))/Ls - rho
    return lenLS_prime

def PathLR(arc1, arc2, al1, rho):
    
    alphaLR = [np.nan, np.nan]
    distLR = [np.nan, np.nan]    
    c_x = arc2.c_x
    c_y = arc2.c_y
    r1 = arc1.arc_radius
    r2 = arc2.arc_radius

    O1 = (r1+rho)*np.array([np.cos(al1), np.sin(al1)])
    C2 = np.array([c_x, c_y])
    len_O1C2 = np.linalg.norm(C2-O1)

    psi1 = np.arctan2(c_y-O1[1], c_x-O1[0])

    cos_psi2 = ((r2-rho)**2+4*rho**2-len_O1C2**2)/(4*rho*(r2-rho))
    cos_psi3 = (len_O1C2**2+4*rho**2-(r2-rho)**2)/(4*rho*len_O1C2)

    if np.abs(cos_psi2) >= 1 or np.abs(cos_psi3) >= 1:
        return np.nan, [np.nan, np.nan]
    
    psi2 = np.arccos(cos_psi2)
    psi3 = np.arccos(cos_psi3)

    phi1 = np.mod(np.pi-al1+psi1-psi3, 2*np.pi)
    phi2 = np.mod(np.pi+psi2, 2*np.pi)
    # phi1 = np.mod(np.pi-al1+psi1+psi3, 2*np.pi)
    # phi2 = np.mod(np.pi-psi2, 2*np.pi)

    return rho*(phi1+phi2), [rho*phi1, rho*phi2]

def PathRL(c_x,c_y, rho, targRadius):
    
    Lcc = np.sqrt(c_x*c_x + (c_y+rho)*(c_y+rho))

    if Lcc>targRadius+3*rho or Lcc < targRadius-rho:
        return [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]
    
    g = (4*rho*rho + Lcc*Lcc - (targRadius+rho)*(targRadius+rho))/(2*Lcc)
    
    psi1 = np.arctan2(c_y+rho, c_x)
    psi2 = np.arccos( g/(2*rho))
    psi3 = np.arccos((Lcc-g)/(targRadius+rho) )
    
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
    
    alphaRL = [alphaRLb, alphaRLa]
    distRL = [distRLb, distRLa]
    segLengths = [[rho*phi1a, rho*phi2a],[rho*phi1b, rho*phi2b]]

    return alphaRL, distRL, segLengths

if __name__ == "__main__":
    
    LSL =0; LSR = 1; RSL = 2; RSR = 3; RLR = 4; LRL = 5;     
    rho = 1    
    pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
    arcfmt = SimpleNamespace(color='m', linewidth=1, linestyle='--', marker='x')
    arrowfmt = SimpleNamespace(color='g', linewidth=1, linestyle='-', marker='x')
    

    # ############################### Test LS #########################
    arc1 = utils.Arc(0,0, 2.5, 0.01, 6.28)
    arc2 = utils.Arc(7., 2.5, 2.5, 0.01, 6.28)
    nd = 10000
    r1 = arc1.arc_radius        
    r2 = arc2.arc_radius  
    psi3 = np.arctan2(arc2.c_y, arc2.c_x)
    pathNum = 1
    alVsLen = np.zeros([nd,4])
    alVec = utils.AngularLinSpace(arc1.angPos_lb, arc1.angPos_ub, nd)
    distVec = np.ones(nd)*np.nan
    lenPrVec = np.ones(nd)*np.nan
    lenPrVec2 = np.ones(nd)*np.nan

    lineSeg = np.array([[arc1.c_x, arc1.c_y], [arc2.c_x, arc2.c_y]])
    
    for indx, al1 in enumerate(alVec):
        
        iniPos = np.array([r1*np.cos(al1), r1*np.sin(al1)])
        iniHdng = al1-np.pi/2
        # c2_trans = np.array([arc2.c_x, arc2.c_y])-iniPos
        # c2_transRot = utils.RotateVec(c2_trans, -iniHdng)
        # alphaRL, distRL, segLengths = PathRL(c2_transRot[0], c2_transRot[1], rho, r2)
        pathLen, segLengths, infPt = PathLS(arc1, arc2, al1, rho)
        
        if np.isfinite(pathLen):            
            # alVsLen[indx,:] = [al1, pathLen, segLengths[0], segLengths[1]]
            distVec[indx] = pathLen
            lenPr = LenLSPrime(arc1, arc2, rho, al1)
            lenPr2 = LenLSPrime2(arc1, arc2, rho, al1)
            lenPrVec[indx] = lenPr
            lenPrVec2[indx] = lenPr2


############################### Test LR #########################
    # arc1 = utils.Arc(0,0, 2.5, 2.5955, 3.33)
    # arc2 = utils.Arc(-2.5, 2, 2.5, 0.01, 6.28)
    # nd = 500
    # r1 = arc1.arc_radius        
    # r2 = arc2.arc_radius  
    
    # pathNum = 0
    
    # alVec = utils.AngularLinSpace(arc1.angPos_lb, arc1.angPos_ub, nd)
    # distVec = np.ones(nd)*np.nan
    # # lenPrVec = np.ones(nd)*np.nan
    # # len_perp_Vec = np.ones(nd)*np.nan
    
    # for indx, al1 in enumerate(alVec):
    # # for indx, al1 in enumerate([.7]):
        
    #     iniPos = np.array([r1*np.cos(al1), r1*np.sin(al1)])
    #     iniHdng = al1-np.pi/2
        
    #     pathLen, segLengths = PathLR(arc1, arc2, al1, rho)
        
    #     if np.isfinite(pathLen):            
    #         # print("pathLen: ", pathLen)
    #         distVec[indx] = pathLen
    #         # du.PlotDubPathSegments([iniPos[0], iniPos[1], iniHdng], 'LR', segLengths[0:2], rho, pathfmt)
    #         # utils.PlotArrow(iniPos, iniHdng, 1, arrowfmt)

            
            
    minIndx = np.argmin(distVec)
    maxIndx = np.argmax(distVec)
    minAl = alVec[minIndx]
    maxAl =alVec[maxIndx]
    print('minAl_num: ', minAl)
    print('maxAl_num: ', maxAl)

    # pathLen, segLengths = PathLS(arc1, arc2, minAl, rho)
    
    pathLen, segLengths, infPt = PathLS(arc1, arc2, minAl, rho)
    # CheckLSMin(arc1, arc2, minAl, rho)
    # print(f"{pathLen=}")
    # print(f"{segLengths=}")
    
    # lenLS_prime_atMin = LenLSPrime(arc1, arc2, rho, minAl)
    # lenLS_prime_atMax = LenLSPrime(arc1, arc2, rho, maxAl)

    # print(f"{lenLS_prime_atMin=}")
    # print(f"{lenLS_prime_atMax=}")
    
    iniPos = np.array([r1*np.cos(minAl), r1*np.sin(minAl)])
    iniHdng = minAl-np.pi/2

    print('segLengths: ', segLengths)
    du.PlotDubPathSegments([iniPos[0], iniPos[1], iniHdng], 'LS', segLengths[0:2], rho, pathfmt)
    utils.PlotArc(arc1, arcfmt)
    utils.PlotArc(arc2, arcfmt)      
    # utils.PlotLineSeg([arc1.c_x, arc1.c_y], [arc2.c_x, arc2.c_y], arrowfmt)

    plt.axis('equal')
    
    # print('alvec: ', alVec)
    # print('distVec: ', distVec)

    plt.figure()
    plt.plot(alVec, distVec)
    plt.plot(alVec, lenPrVec)
    plt.plot(alVec, lenPrVec2, linestyle='--')
    plt.vlines(minAl, 0, 18)
    plt.vlines(maxAl, 0, 18)


    plt.show()