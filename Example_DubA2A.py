from timeit import default_timer as timer
import DubinsA2A as da2a
    
tic = timer()
arc1 = da2a.Arc(1, 2, 2.6, 2.5, 6.28)
arc2 = da2a.Arc(-3.5, -3., 2.8, .5, 4.28)

A2ADub = da2a.Arc2ArcDubins(arc1, arc2, 1) 


minLength, minPath, candPathsList = A2ADub.A2AMinDubins()   
comp_time = timer()-tic

print('minLength: ', minLength)  
print('comp_time: ', comp_time) 

if minPath:    
    A2ADub.PlotAllPaths(candPathsList)
    print('minPath: ', minPath)  
    A2ADub.PlotA2APath((minPath.angPos_arc1, minPath.angPos_arc2), minPath.segLengths, minPath.pathType)
    