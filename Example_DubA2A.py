from timeit import default_timer as timer
from types import SimpleNamespace
import utils
import DubinsA2A as da2a
import DubinsA2A_Num as da2aNum
import matplotlib.pyplot as plt
    

arc1 = utils.Arc(1, 2, 2.6, 1.5, 5.9) # arguments are arc1_cntr_x, arc1_cntr_y, arc1_radius, arc1_lowerLimit (angular Pos), arc1_upperLimit
arc2 = utils.Arc(3.5, -5., 2.8, .8, 4.28) #similar arguments for arc2
rho = 1 #minimum turn radius


## computing shortest arc to arc dubins path using analytical results
tic = timer()
A2ADub = da2a.Arc2ArcDubins(arc1, arc2, rho) 
minLength, minPath, candPathsList = A2ADub.A2AMinDubins()   #This returns the length of the minimum path, minimum path, and all the candidate paths
comp_time = timer()-tic

print('Length of the shortest arc to arc Dubins path: \n', minLength)  
# print('Parameters of the min Path: \n', minPath) 
print('Computation time of the analytical solution: \n', comp_time) 


## computing shortest arc to arc dubins path using numerical methods (brute force discretizing)
tic = timer()
num_disc = 1000 #number of discrete samples on eacha arc
minPathLengthNum, sp = da2aNum.A2AMinDubinsNum(arc1, arc2, rho, num_disc)
comp_time_num = timer()-tic
print('Length of the shortest arc to arc Dubins path (numerical): \n', minPathLengthNum)  
# print('Parameters of the min Path: \n', sp) 
print('Computation time of the numerical solution: \n', comp_time_num) 

## uncomment the lines below to see the plots of the candidate paths
# if minPath:    
#     A2ADub.PlotAllPaths(candPathsList) #This plots all the candidate paths
#     print('minPath: ', minPath)  #This prints the parameters of the minimum path

## uncomment the lines below to see the plots of the minimum paths
if minPath:    
    A2ADub.PlotA2APath((minPath.angPos_arc1, minPath.angPos_arc2), minPath.segLengths, minPath.pathType) #This plots the minimum path
    pathfmt = SimpleNamespace(color='m', linewidth=2, linestyle='--', marker='x')    
    A2ADub.PlotA2APath((sp.angPos_arc1, sp.angPos_arc2), sp.segLengths, sp.pathType,arc1, arc2, pathfmt) #This plots the minimum path
    plt.show()
    
