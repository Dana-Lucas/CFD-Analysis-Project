# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:38:29 2020

@author: Dana
"""
import numpy as np
import matplotlib.pyplot as plt

'''
This portion shows a quiver plot showing the magnitude of the 
velocity at certain coordinates where the plates are stationary
'''
# Stationary plates
def velocity_laminarflowStationary(l,v):
    x,y = np.meshgrid(np.arange(0,l,10),np.arange(0,l,10))

    vvals = []
    # Determine what the velocity will be at that particular y location
    # This comes from an equation described in the FLuid Mechanics textbook
    # that describes a velocity profile given that there is laminar, fully
    # developed flow.
    for i in np.arange(0,l,10):
        if i < l/2:
            r = l/2 - i
        elif i > l/2:
            r = i - l/2
        else:
            r = 0
        i = v*(1-(r**2/(l/2)**2))
        vvals.append(i)

    vxvals, vyvals = np.meshgrid(np.arange(0,l,10),vvals)
    plt.figure(dpi=100)
    
    # plot x and y and the magnitude of velocity of each particle
    plt.quiver(x,y,vyvals,0) 
                        
    plt.title('Magnitude of Velocity (fluid flow left to right at {} m/s)'.format(v))
    plt.show()
    return

print(velocity_laminarflowStationary(100,20))


'''
This portion shows a quiver plot showing the magnitude of the 
velocity at certain coordinates where the top plate is moving
'''
# The find the velocity at any x-y position in a fluid

# Create a plot of a specific plate velocity and distance between plates
def velocity_laminarflow(l,v):
    x,y = np.meshgrid(np.arange(0,l,10),np.arange(0,l,10))
    
    # Equation from fluid mechanics textbook to describe velocity of particle, u, when
    # velocity of plate, v, y-location between plates, y, and distance between plates, l, are known
    u = y*v/l 
              
    # Color coordinate the quiver plot, where the particles with larger magnitude have a darker color
    # and the particles with smaller velocity are colored yellow
    color_array = np.sqrt(((u-l)/2)**2 + ((0-l)/2)**2)
    
    plt.figure(dpi=100)
    
    # plot x and y where the magnitudes defined by the value of u and each row looks the same 
    # since there is no acceleration of the particles or pressure pushing it forward
    plt.quiver(x,y,u,0,color_array) 
    
    plt.title('Magnitude of Velocity (fluid flow left to right, top plate moving at {} m/s)'.format(v))
    plt.show()
    return u

print(velocity_laminarflow(100,20))

'''
This creates a dictionary to use as reference for what the particle velocity will be at the desired location between
plates moving at the desired speed
'''
# Create a dictionary of velocity at different y-locations of plate velocities from 0-50
platevelocity = []
l = 100
ilist = []
for i in range(51): # Look at 51 different plate velocities from 0 to 51
    platevelocity.append(i)
    velocity = []
    for j in range(101): # Look at each of the y-positions for each of those plate velocities
        velocity.append(j*i/l)
    ilist.append(velocity)
#dictionary = dict(zip(list(range(51)),ilist))
#print(dictionary[32][25]) # shows the velocity of a particle 3/4 the way from the moving plate that is traveling at 32 units/time step