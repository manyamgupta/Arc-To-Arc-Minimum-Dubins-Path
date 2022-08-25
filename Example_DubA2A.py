from timeit import default_timer as timer
import DubinsA2A as da2a
    
tic = timer()
arc1 = da2a.Arc(1, 2, 2.6, 2.5, 6.28) # arguments are arc1_cntr_x, arc1_cntr_y, arc1_radius, arc1_lowerLimit (angular Pos), arc1_upperLimit
arc2 = da2a.Arc(-3.5, -3., 2.8, .5, 4.28) #similar arguments for arc2

A2ADub = da2a.Arc2ArcDubins(arc1, arc2, 1) 


minLength, minPath, candPathsList = A2ADub.A2AMinDubins()   #This returns the length of the minimum path, minimum path, and all the candidate paths
comp_time = timer()-tic

print('minLength: ', minLength)  
print('comp_time: ', comp_time) 

if minPath:    
    A2ADub.PlotAllPaths(candPathsList) #This plots all the candidate paths
    print('minPath: ', minPath)  #This prints the parameters of the minimum path
    A2ADub.PlotA2APath((minPath.angPos_arc1, minPath.angPos_arc2), minPath.segLengths, minPath.pathType) #This plots the minimum path
    