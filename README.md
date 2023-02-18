# Python_N_body_simulation
2D particles gravity simulation using the Barnes-Hut algorithm (quadtree, o(nln(n)) and Cython


![](7824_Theta_0.25.gif)


To use only Python comment computeFast and uncomment computeSlow in main.py as below:
from computeSlow import simulator
#from computeFast import simulator

To improve speed using Cython comment computeSlow and uncomment computeFast in main.py as below:
#from computeSlow import simulator
from computeFast import simulator
Also run this: python setup.py build_ext --inplace


In main.py, the variable 'theta' define the accuracy of the simulation
The lower theta is, the more accurate the simulation will be but at the cost of performance, a pretty good compromise is 0.5
