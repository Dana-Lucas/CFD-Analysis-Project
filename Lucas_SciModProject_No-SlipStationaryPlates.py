# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:01:54 2020

@author: Dana
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation 

'''
This creates a simulation of particles moving between a pair of 'plates' in two different scenarios,
the first where both plates are stationary and the second where the bottom plate is moving at 10 
units per time step in addition to the pressure forcing the fluid throughat 1 unit per timestep
'''

# Constants and lists similar to both simulations
N = 100
t_max = 100
update_order = []

# Create update function for the animation
def update(i):
    ax1.cla()
    ax1.imshow(A[i],vmin = 0, vmax = 1, cmap = 'tab20c')  
    ax1.axhline(linewidth=5,c='k')
    ax1.axhline(y=N-1,linewidth=4,c='k')

for i in range(N):
    for j in range(200):
        update_order.append((i,j))

# The update order is reversed to ensure particles (which always travel left to right)
# don't move multiple times during a single update
update_order = update_order[::-1] 

A = np.zeros((1,N,200),dtype=np.int8)

# velocity at any y location, given laminar, fully developed flow
# The r in this equation is the radius from the center of the 'pipe', 
# so this is rewriting the velocity in terms of the y-location, i
def velocityProfile(i):
    global N
    if i < (N/2):
        r = (N/2) - i
    elif i > (N/2):
        r = i - (N/2)
    else:
        r = 0
    return 10*(1-(r**2/(N/2)**2))

# Certain boxes in the left most column are colored differently to signify particles that will be tracked
for i in range(N):
    # Only draw every tenth particle so individual particles can be seen
    if i%10==0:
        A[0,i,0] = 1
      
t = 0        
while t<19: #A[-1,50,N-1] == 0: # A[-1,i,N-1] == 0: With no slip condition, some particles will never move, so tracking when the last one passes will not work any more
    A_layer = np.zeros((1,N,200),dtype=np.int8)  # set up array
    A_layer[0] = A[-1]                         # make it a duplicate of current
    A = np.concatenate((A,A_layer))            # perform stack

    for i,j in update_order:
        # Get the velocity of that particle
        vparticle = int(velocityProfile(i)//1)

        # If the particle is one we are tracking, then update it
        if A[-1,i,j] == 1:
            A[-1,i,(j+vparticle)%200] = 1
            A[-1,i,j] = 0
    t+=1
#        if i==50:
#            try: 
#                A[-1,i,(j+vparticle)] = 1
#                A[-1,i,j] = 0
#            except:
#                A[-1,i,j] = 0
#        else:
#            if A[-1,i,j] == 1:
#                print(time.time())
#                A[-1,i,(j+vparticle)%200] = 1
#                A[-1,i,j] = 0
            
# Actually create the plots for this animation
fig = plt.figure(dpi=100)
  
ax1 = fig.add_subplot(111)
plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
plt.title('Both Plates Stationary')

ani = animation.FuncAnimation(fig, update, frames=t_max, interval=10, repeat_delay=50)

plt.show()