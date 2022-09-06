import numpy as np
import os
import dubins
import utils
import dubutils as du
import matplotlib.pyplot as plt
import DubinsA2A as da2a
from types import SimpleNamespace
import time
def write_tsp_file(edge_weights, filename):

    N = np.shape(edge_weights)[0]
    with open(filename, 'w') as f:
        f.writelines('NAME: Symmetric TSP Weights \n')
        f.writelines('COMMENT: concorde routine for the atsp \n')
        f.writelines('TYPE: TSP \n')
        f.writelines('DIMENSION: '+str(N)+'\n' )
        f.writelines('EDGE_WEIGHT_TYPE: EXPLICIT \n')
        f.writelines('EDGE_WEIGHT_FORMAT: FULL_MATRIX \n')
        f.writelines('EDGE_WEIGHT_SECTION \n')
        for i in range(N):
            for j in range(N):                
                f.writelines(str(int(edge_weights[i,j]))+' ')
            f.writelines('\n')            
        f.writelines('EOF')
          
    return

def Trans3N_Atsp2Sym(atspWts):
    N = np.shape(atspWts)[0]    
    inf_diag_zeros = infn*((np.ones((N,N))) - np.diag(np.ones(N)))
    
    sym_wts1 = np.concatenate((infn*np.ones((N,N)), inf_diag_zeros, np.transpose(atspWts)), axis=1)
    sym_wts2 = np.concatenate((inf_diag_zeros, infn*np.ones((N,N)), inf_diag_zeros), axis=1)
    sym_wts3 = np.concatenate((atspWts, inf_diag_zeros, infn*np.ones((N,N))), axis=1)
    
    sym_wts = np.vstack((sym_wts1, sym_wts2, sym_wts3))
    
    return sym_wts
def ReadConcSol(filename):
    with open(filename) as f:
        lines = f.readlines()
    tsp_tour=[]
    for i in range(1, len(lines)):
        l = lines[i].split()
        for j in range(len(l)):
            tsp_tour.append(int(l[j]))
            
    return tsp_tour

def Concorde_Atsp(atspWts):
    
    sym_weights = Trans3N_Atsp2Sym(atspWts)
    # print(f"{atspWts=} ")
    # print(f"{sym_weights=} ")
    write_tsp_file(sym_weights, 'sym_weights.tsp')
    os.system("./concorde sym_weights.tsp >> concorde_output.txt")
    opt_tour = ReadConcSol('sym_weights.sol')

    # tourCost_3N = TourCost(opt_tour, sym_weights)
    # PrintTour(opt_tour, sym_weights)
    # print(f"{tourCost_3N=} ")
    
    opt_tour_asym = opt_tour[0::3]
    
    opt_tour_asym_rev =  np.concatenate(([opt_tour_asym[0]], opt_tour_asym[1:][::-1]))
    if TourCost(opt_tour_asym_rev, atspWts) < TourCost(opt_tour_asym, atspWts):
        opt_tour_asym = opt_tour_asym_rev
    tourCost_asym =  TourCost(opt_tour_asym, atspWts)
    # PrintTour(opt_tour_asym, atspWts)
    # print(f"{tourCost_asym=} ")
    
    return opt_tour_asym

def NoonBeanTrans(cost_orig, setsSizes):
    ## setsSizes: array, each entry is the size of the corresponding set
 
    setsSizes = setsSizes.astype(int)
    no_sets = len(setsSizes)
    cost_trans = np.copy(cost_orig)
    nT_tr = np.shape(cost_orig)[0]
    ## Cycle of zero cost edges with in a set
    j=0
    for i in range(no_sets):        
        if setsSizes[i] > 1:            
            for ind in range(int(setsSizes[i]-1)):
                cost_trans[j+ind,j+ind+1] = 0
                cost_trans[j+ind+1,j+ind] = infn            
            cost_trans[j+setsSizes[i]-1, j] = 0
            cost_trans[j, j+setsSizes[i]-1] = infn    
        j=j+setsSizes[i]
    
    ## replacing edges going out of a set with succesors
    cumsum_setsizes = np.cumsum(setsSizes)
    cumsum_setsizes = np.insert(cumsum_setsizes, 0, 0)    
    for i in range(no_sets):
        
        j = cumsum_setsizes[i]
        if setsSizes[i] > 1:
            
            for k in range(setsSizes[i]):
                
                inds_of_set = np.arange(cumsum_setsizes[i], cumsum_setsizes[i+1])
                inds_of_set_pred = np.roll(inds_of_set, -1)
                ind_out = np.concatenate( (np.arange(0, inds_of_set[0]), np.arange(inds_of_set[-1]+1, nT_tr)) )
                cur_ind  = j+k
                cur_ind_pred = inds_of_set_pred[k]
                cost_trans[cur_ind,ind_out] =  cost_orig[cur_ind_pred, ind_out]+ bigM           
        
        elif setsSizes[i] == 1:
            
            ind_out = np.concatenate( (np.arange(0, j), np.arange(j+1, nT_tr)) )
            cur_ind  = j          
            cost_trans[cur_ind,ind_out] =  cost_orig[cur_ind, ind_out]+ bigM

    return cost_trans

def GenerateDubDistMat(targ_coords, ndisc, rho):
    
    N = np.shape(targ_coords)[0]
    cost_orig = np.zeros((N*ndisc, N*ndisc))
    headingsVec = np.transpose(np.matrix(np.linspace(0, 2*np.pi, ndisc+1)[0:-1]))
    allConfigs = np.empty((0,3))
    for i in range(N):        
        cur_configs = np.concatenate((targ_coords[i,0]*np.ones((ndisc, 1)), targ_coords[i,1]*np.ones((ndisc, 1)), headingsVec), axis=1)
        allConfigs = np.concatenate((allConfigs, cur_configs), axis=0)
    nConfs = np.shape(allConfigs)[0]
    # print(f"{allConfigs=} ")
    
    for i in range(nConfs):
        for j in range(nConfs):
            
            frmConf = np.array(allConfigs[i,:])[0]
            toConf = np.array(allConfigs[j,:])[0]
            dubPath = dubins.shortest_path(frmConf, toConf, rho)
            cost_orig[i,j] = dubPath.path_length()
            
    return cost_orig, allConfigs

def GenODTSPDistMat(targ_coords, ndisc, targRadius, rho):
    
    N = np.shape(targ_coords)[0]
    cost_orig = np.zeros((N*ndisc, N*ndisc))
    alVec = np.linspace(0, 2*np.pi, ndisc+1)[0:-1]
    allConfigs = np.empty((0,3))
    for i in range(N):        
        
        xVec = targ_coords[i,0]+targRadius*np.cos(alVec)
        yVec = targ_coords[i,1]+targRadius*np.sin(alVec)
        tVec = alVec-np.pi/2
        cur_configs = np.transpose(np.matrix([xVec, yVec, tVec]))        
        allConfigs = np.concatenate((allConfigs, cur_configs), axis=0)
    nConfs = np.shape(allConfigs)[0]
    # print(f"{allConfigs=} ")
    
    for i in range(nConfs):
        for j in range(nConfs):
            
            frmConf = np.array(allConfigs[i,:])[0]
            toConf = np.array(allConfigs[j,:])[0]
            dubPath = dubins.shortest_path(frmConf, toConf, rho)
            cost_orig[i,j] = int(np.round(dubPath.path_length()))
            
    return cost_orig, allConfigs

def ODTSPLB_DistMat(targ_coords, ndisc, targRadius, rho):
    
    N = np.shape(targ_coords)[0]
    cost_orig = np.zeros((N*ndisc, N*ndisc))
    alVec = np.linspace(0, 2*np.pi, ndisc+1)
    allArcs = np.empty((0,5))
    for i in range(N):        
        
        xVec = targ_coords[i,0]*np.ones(ndisc)
        yVec = targ_coords[i,1]*np.ones(ndisc)
        radVec = targRadius*np.ones(ndisc)
        al_lVec = alVec[0:ndisc]
        al_uVec = alVec[1:ndisc+1]
        
        cur_Arcs = np.transpose(np.matrix([xVec, yVec, radVec,al_lVec, al_uVec ]))        
        allArcs = np.concatenate((allArcs, cur_Arcs), axis=0)
    nArcs = np.shape(allArcs)[0]
    # print(f"{allArcs=} ")
    
    for i in range(nArcs):
        for j in range(nArcs):            
            frmArc = np.array(allArcs[i,:])[0]
            toArc = np.array(allArcs[j,:])[0]
            if np.linalg.norm(frmArc[0:2]-toArc[0:2])>0.00001:
                arc1 = utils.Arc(frmArc[0], frmArc[1], frmArc[2], frmArc[3], frmArc[4]) # arguments are arc1_cntr_x, arc1_cntr_y, arc1_radius, arc1_lowerLimit (angular Pos), arc1_upperLimit
                arc2 = utils.Arc(toArc[0], toArc[1], toArc[2], toArc[3], toArc[4]) #
                A2ADub = da2a.Arc2ArcDubins(arc1, arc2, rho)
                minLength, minPath, candPathsList = A2ADub.A2AMinDubins()
                
                cost_orig[i,j] = int(np.round(minLength))
            else:
                cost_orig[i,j] = bigM
            
    return cost_orig, allArcs


def TourCost(tour, costMat):
    tourCost = costMat[tour[-1], tour[0]]
    for i in range(len(tour)-1):    
        tourCost += costMat[tour[i], tour[i+1]]
    
    return tourCost
def PrintTour(tour, costMat):
    
    for i in range(len(tour)-1):    
        print(str(tour[i])+'-->'+str(tour[i+1])+': '+str(costMat[tour[i], tour[i+1]]))
    print(str(tour[-1])+'-->'+str(tour[0])+': '+str(costMat[tour[-1], tour[0]]))
    
    return
def ExtractDubTour(fullTour, ndisc, cost_trans):
    fullTour = np.array(fullTour)
    no_confs = len(fullTour)
    if cost_trans[fullTour[-1], fullTour[0]] != 0:        
        inds = range(0,no_confs, ndisc)        
    else:
        i=0
        while(cost_trans[fullTour[i],fullTour[i+1]] == 0):
            i=i+1
        inds = range(i+1, no_confs, ndisc)
        
    oneSetTour = fullTour[inds]
    return oneSetTour

def TestDubTSP(N):
    
    ndisc = 4
    rho=50
    targ_coords = np.random.randint(1000,size=(N,2)) 
    print(f"{targ_coords=} ")
       
    sets_sizes = int(ndisc)*np.ones(N).astype(int)
    cost_orig, allConfigs = GenerateDubDistMat(targ_coords, ndisc, rho)
    cost_trans = NoonBeanTrans(cost_orig, sets_sizes)
    opt_tour_asym = Concorde_Atsp(cost_trans)
    if TourCost(opt_tour_asym, cost_trans) > TourCost(np.flipud(opt_tour_asym), cost_trans):
        opt_tour_asym = np.flipud(opt_tour_asym)
        
    print(f"{opt_tour_asym=} ")
    
    plt.scatter(targ_coords[:,0], targ_coords[:,1], marker='x',color='k')
    oneSetTour = ExtractDubTour(opt_tour_asym, ndisc, cost_trans)
    PlotDubTour(oneSetTour, allConfigs, rho)
    plt.show()
    return

def ODTSP_SmplSoln(targ_coords, targRadius, rho, ndisc):
    no_targs = np.shape(targ_coords)[0]
    sets_sizes = int(ndisc)*np.ones(no_targs).astype(int)
    cost_orig, allConfigs = GenODTSPDistMat(targ_coords, ndisc, targRadius, rho)
    cost_trans = NoonBeanTrans(cost_orig, sets_sizes)
    opt_tour_asym = Concorde_Atsp(cost_trans)
    if TourCost(opt_tour_asym, cost_trans) > TourCost(np.flipud(opt_tour_asym), cost_trans):
        opt_tour_asym = np.flipud(opt_tour_asym)
    oneSetTour = ExtractDubTour(opt_tour_asym, ndisc, cost_trans)
    cost_ODTSP =TourCost(oneSetTour, cost_orig)
    # print(f"{opt_tour_asym=} ")
    # print(f"{cost_ODTSP=} ")
    
    # cirFmt = SimpleNamespace(color='c', linewidth=2, linestyle='-', marker='x')
    # plt.scatter(targ_coords[:,0], targ_coords[:,1], marker='x',color='k')
    # for i in range(N):
    #     utils.PlotCircle(targ_coords[i,:], targRadius, cirFmt)
    
    # PlotDubTour(oneSetTour, allConfigs, rho)
    # plt.show()
    return cost_ODTSP, oneSetTour, allConfigs

def ODTSP_LB(targ_coords, targRadius, rho, ndisc):
    no_targs = np.shape(targ_coords)[0]
    sets_sizes = int(ndisc)*np.ones(no_targs).astype(int)
    cost_orig, allArcs = ODTSPLB_DistMat(targ_coords, ndisc, targRadius, rho)
    cost_trans = NoonBeanTrans(cost_orig, sets_sizes)
    opt_tour_asym = Concorde_Atsp(cost_trans)
    if TourCost(opt_tour_asym, cost_trans) > TourCost(np.flipud(opt_tour_asym), cost_trans):
        opt_tour_asym = np.flipud(opt_tour_asym)
    oneSetTour = ExtractDubTour(opt_tour_asym, ndisc, cost_trans)
    cost_ODTSP_LB =TourCost(oneSetTour, cost_orig)
    # print(f"{opt_tour_asym=} ")
    # print(f"{cost_ODTSP_LB=} ")
    
    # cirFmt = SimpleNamespace(color='c', linewidth=2, linestyle='-', marker='x')
    # plt.scatter(targ_coords[:,0], targ_coords[:,1], marker='x',color='k')
    # for i in range(N):
    #     utils.PlotCircle(targ_coords[i,:], targRadius, cirFmt)
    
    # PlotDubTour(oneSetTour, allConfigs, rho)
    # plt.show()
    return cost_ODTSP_LB, oneSetTour, allArcs

def PlotOrbits(targ_coords, targRadius):
    no_targs = np.shape(targ_coords)[0]
    cirFmt = SimpleNamespace(color='c', linewidth=2, linestyle='-', marker='x')
    plt.scatter(targ_coords[:,0], targ_coords[:,1], marker='x',color='k')
    for i in range(no_targs):
        utils.PlotCircle(targ_coords[i,:], targRadius, cirFmt)
def PlotDubTour(opt_tour, allConfigs, rho, pathfmt):
    
    opt_tour = np.append(opt_tour, opt_tour[0])
    for i in range(len(opt_tour)-1):
        fromConf = np.array(allConfigs[opt_tour[i]])[0]
        toConf = np.array(allConfigs[opt_tour[i+1]])[0]
        dubPath = dubins.shortest_path(fromConf, toConf, rho)    
        du.PlotDubinsPath(dubPath, pathfmt)
    plt.axis('equal')
    # plt.show()
    return
    
def PlotODTSP_LBTour(ODTSP_LB_Tour, allArcs, rho, pathfmt):
    
    ODTSP_LB_Tour = np.append(ODTSP_LB_Tour, ODTSP_LB_Tour[0])
    for i in range(len(ODTSP_LB_Tour)-1):
        frmArc = np.array(allArcs[ODTSP_LB_Tour[i],:])[0]
        toArc = np.array(allArcs[ODTSP_LB_Tour[i+1],:])[0]
        arc1 = utils.Arc(frmArc[0], frmArc[1], frmArc[2], frmArc[3], frmArc[4]) # 
        arc2 = utils.Arc(toArc[0], toArc[1], toArc[2], toArc[3], toArc[4]) #
        A2ADub = da2a.Arc2ArcDubins(arc1, arc2, rho)
        minLength, minPath, candPathsList = A2ADub.A2AMinDubins()
        A2ADub.PlotA2APath((minPath.angPos_arc1, minPath.angPos_arc2), minPath.segLengths, minPath.pathType, arc1, arc2, pathfmt)
    
    return
def FeasSolnODTSP(ODTSP_LB_Tour, allArcs, rho, ndisc, plotflag):
    
    no_targs = len(ODTSP_LB_Tour)
    ODTSP_LB_Tour = np.append(ODTSP_LB_Tour, ODTSP_LB_Tour[0])
    angPos_incm = np.zeros(no_targs)
    angPos_outg = np.zeros(no_targs)
    angPosFeas = np.zeros(no_targs)
    seqArcs = []
    for i in range(no_targs):
        # from_targ_num = TargNum(ODTSP_LB_Tour[i], ndisc)
        # to_targ_num = TargNum(ODTSP_LB_Tour[i+1], ndisc)
        j = np.mod(i+1, no_targs)
        frmArc = np.array(allArcs[ODTSP_LB_Tour[i],:])[0]
        toArc = np.array(allArcs[ODTSP_LB_Tour[j],:])[0]
        arc1 = utils.Arc(frmArc[0], frmArc[1], frmArc[2], frmArc[3], frmArc[4]) # 
        arc2 = utils.Arc(toArc[0], toArc[1], toArc[2], toArc[3], toArc[4]) #
        A2ADub = da2a.Arc2ArcDubins(arc1, arc2, rho)
        minLength, minPath, candPathsList = A2ADub.A2AMinDubins()
        angPos_outg[i] = minPath.angPos_arc1
        angPos_incm[j] = minPath.angPos_arc2
        seqArcs.append(arc1)
        
        
    for i in range(no_targs):
        
        midAl = utils.MidAng(angPos_outg[i], angPos_incm[i])
        if utils.InInt(seqArcs[i].angPos_lb, seqArcs[i].angPos_ub, midAl):
            angPosFeas[i] = midAl
        else:
            angPosFeas[i] = utils.MidAng(angPos_incm[i], angPos_outg[i])
    
    # print(f"{seqArcs=}")    
    # print(f"{angPos_incm=}")
    # print(f"{angPos_outg=}")
    # print(f"{angPosFeas=}")
    # plotflag = True
    cost_feas_ODTSP = ComputeCostTour(seqArcs, angPosFeas, rho, plotflag)
    
    return cost_feas_ODTSP, seqArcs, angPosFeas
def ComputeCostTour(seqArcs, angPosFeas, rho, plotflag):

    pathLength = 0
    no_targs = len(seqArcs)
    pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')    
    
    for i in range(no_targs):
        j = np.mod(i+1, no_targs)    
        frmArc = seqArcs[i]
        toArc = seqArcs[j]
        al_i = angPosFeas[i]
        al_j = angPosFeas[j]
        frmConf = [frmArc.cntr_x+frmArc.arc_radius*np.cos(al_i), frmArc.cntr_y+frmArc.arc_radius*np.sin(al_i), al_i-np.pi/2]
        toConf = [toArc.cntr_x+toArc.arc_radius*np.cos(al_j), toArc.cntr_y+toArc.arc_radius*np.sin(al_j), al_j-np.pi/2]
        dubPath = dubins.shortest_path(frmConf, toConf, rho)
        pathLength += dubPath.path_length()
        if plotflag:
            du.PlotDubinsPath(dubPath, pathfmt)
    return pathLength

def TargNum(odtsp_no, ndisc):
    
    return int(np.floor(odtsp_no/ndisc))

if __name__ == "__main__":
    bigM = 10000
    infn = 1000000
    # cost_orig = np.random.randint(20, size=(9,9))
    # cost_orig = np.array([[ 5, 19, 16,  8, 17, 18,  0, 17,  9],
    #    [ 4,  3,  9,  3,  3, 16, 15,  7,  9],
    #    [ 2,  0,  0, 14,  2, 15, 19,  7, 12],
    #    [12, 11, 11,  9,  8, 15, 17,  6,  0],
    #    [ 3,  9,  3,  1,  9, 18,  6, 12, 17],
    #    [ 6, 18, 18,  4,  6,  2,  6,  5,  7],
    #    [ 1, 15, 11, 19, 15,  3, 11, 16,  7],
    #    [ 0,  5,  4,  8,  6,  1, 14,  9,  5],
    #    [13, 10,  9, 12,  8, 15, 12,  7, 15]])
    # sets_sizes = np.array([4,1,4])
    # cost_trans = NoonBeanTrans(cost_orig, sets_sizes)
    # print(f"{cost_orig=} ")
    # print(f"{cost_trans=} ")
    # TestDubTSP(5)
    
    ndisc = 8
    rho=50
    targRadius = 60
    N = 10
    targ_coords = np.random.randint(1000,size=(N,2)) 
    # targ_coords = np.array([[376, 315],
    #    [622, 951],
    #    [502, 347],
    #    [922, 542],
    #    [164, 644],
    #    [193, 314],
    #    [786, 561],
    #    [348, 120],
    #    [363, 534],
    #    [958, 609]])
    # targ_coords = np.array([[996, 916],
    #    [441,  94],
    #    [232, 798],
    #    [305, 596],
    #    [894, 323]])
    print(f"{targ_coords=} ")
    # cost_ODTSP, ODTSP_Tour, allConfigs = ODTSP_SmplSoln(targ_coords, targRadius, rho, ndisc)
    # print(f"{cost_ODTSP=} ")
    
    tic = time.time()
    cost_ODTSP_LB, ODTSP_LB_Tour, allArcs = ODTSP_LB(targ_coords, targRadius, rho, ndisc)    
    print(f"{cost_ODTSP_LB=} ")  
    
    compt_time_lb = time.time()-tic
    print(f"{compt_time_lb=} ")  
    
    plotFeasSoln = True
    cost_feas_ODTSP, seqArcs, angPosFeas = FeasSolnODTSP(ODTSP_LB_Tour, allArcs, rho, ndisc, plotFeasSoln)
    print(f"{cost_feas_ODTSP=} ")  
    
    PlotOrbits(targ_coords, targRadius)
    pathfmt = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')    
    # PlotDubTour(ODTSP_Tour, allConfigs, rho, pathfmt)
    pathfmt = SimpleNamespace(color='green', linewidth=2, linestyle='-', marker='x')    
    PlotODTSP_LBTour(ODTSP_LB_Tour, allArcs, rho, pathfmt)
    plt.show()

        