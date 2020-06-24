# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:54:59 2020

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
    ax2.cla()
    ax2.imshow(B[i],vmin = 0, vmax = 1, cmap = 'tab20c') 
    ax2.axhline(linewidth=5,c='k')
    ax2.axhline(y=N-1,linewidth=5,c='k')

for i in range(N):
    for j in range(N):
        update_order.append((i,j))

# The update order is reversed to ensure particles (which always travel left to right)
# don't move multiple times during a single update
update_order = update_order[::-1] 

#######
# Start of code specific to two stationary plates; added to subplot 121

A = np.zeros((1,N,N),dtype=np.int8)

# velocity at any y location, given laminar, fully developed flow
def velocityProfile(i):
    global N
    if i < (N/2):
        r = (N/2) - i
    elif i > (N/2):
        r = i - (N/2)
    else:
        r = 0
    return 5*(1-(r**2/(N/2)**2))

# Certain boxes in the left most column are colored differently to signify particles that will be tracked
for i in range(N):
    # Only draw every tenth particle so individual particles can be seen
    if i%10==0:
        A[0,i,0] = 1
        
while A[-1,i,N-1] == 0:
    A_layer = np.zeros((1,N,N),dtype=np.int8)  # set up array
    A_layer[0] = A[-1]                         # make it a duplicate of current
    A = np.concatenate((A,A_layer))            # perform stack

    for i,j in update_order:
        # When no plates are moving, the particles all travel at the same speed, 
        # so the box to the right of the colored box is colored and the colored box 
        # becomes colored no more
        if A[-1,i,j] == 1:
            A[-1,i,(j+1)%N] = 1
            A[-1,i,j] = 0

######
# Start of code specific to one moving plate; added to subplot 122
 
B = np.zeros((1,N,N),dtype=np.int8)

# Certain boxes in the left most column are colored differently to signify particles that will be tracked
for i in range(N):
    # Only draw every tenth particle so individual particles can be seen
    if i%10==0:
        B[0,i,0] = 1

while B[-1,i,N-1] == 0:
    B_layer = np.zeros((1,N,N),dtype=np.int8)
    B_layer[0] = B[-1]
    B = np.concatenate((B,B_layer))

    for i,j in update_order:
        if B[-1,i,j] == 1:
            # This try-except statement just prevents particles that have already reached the 
            # edge from crashing the program as they try to move further. It basically
            # just tells them to stop at the end
            try: 
                B[-1,i,int(j+1+i*10/N)] = 1
                B[-1,i,j] = 0
            except:
                B[-1,i,j] = 0

######
# Actually create the subplots for these simulations, side by side for easy comparison
fig = plt.figure(dpi=100)
  
ax1 = fig.add_subplot(121)
plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
plt.title('Both Plates Stationary')

ax2 = fig.add_subplot(122)
plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
plt.title('Bottom Plate Moving')

ani = animation.FuncAnimation(fig, update, frames=t_max, interval=10, repeat_delay=50)

plt.show()