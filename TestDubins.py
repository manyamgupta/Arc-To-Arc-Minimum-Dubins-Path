import dubins
import matplotlib.pyplot as plt
import dubutils as du
from collections import namedtuple
import time
from types import SimpleNamespace

plotformat = SimpleNamespace(color='blue', linewidth=2, linestyle='-', marker='x')
tic = time.time()
dubPath = dubins.path([0,0,.75], [.1,.5,2.75],1,5)
# dubPath = dubins.shortest_path([0,0,.75], [.1,.5,2.75],0)
comp_time = time.time()-tic
print(f"{comp_time=}")
du.PlotDubinsPath(dubPath, plotformat)
plt.axis('equal')
plt.show()

