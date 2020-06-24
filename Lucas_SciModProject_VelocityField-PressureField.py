# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:30:54 2020

@author: Dana
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Make x and y be identified as variables rather than values
x,y = sp.symbols('x y')

####
# EXAMPLE 1
# These are the x and y components of the vector field
u = 3*x+5
v = -2*y+7*x

####

#####
## EXAMPLE 2
## These are the x and y components of the vector field
#u = 3*x**2*y
#v = 4*x+3*y
#####

# Constant for density of water (will vary depending on the fluid that's being modeled)
rho = 1000

# Derivative of x component of pressure field (Derived using N-S equation)
dPdx=rho*(u*sp.diff(u,x)+v*sp.diff(u,y))

# Derivative of y component of pressure field (Derived using N-S equation)
dPdy=rho*(u*sp.diff(v,x)+v*sp.diff(v,y))

# x component of pressure field
Px = sp.integrate(dPdx,x)

# y component of pressure field
Py = sp.integrate(dPdy,y)

# Total pressure field
P = Px + Py

print(P)
print(Px)
print(Py)

# Create the grid that the fields will be graphed in
N = 5 # range of values
n = 100 # kinda like the density of values in between each integer value; a higher value gives a smoother plot
z = np.zeros((2*n*N,2*n*N))
zv = z.copy()

# x and y values for the color mesh plots (more values)
xvals = np.arange(-N,N,1/n)
yvals = xvals.copy()

# these are the smaller arrays that cont up by integer values; this is used for the quiver plots so they don't look crowded
zx = np.zeros((2*N,2*N))
zy = zx.copy()
zvx = zx.copy()
zvy = zx.copy()

# just the x and y values counting up by integers (for the quiver plots)
xvalsc = np.arange(-N,N,1)
yvalsc = xvalsc.copy()

####
# EXAMPLE 1
# Actually substitute x/y position values for the velocity and pressure fields found above 
# (must manually input pressure field b/c the system has it in terms of variables x, y above that you can't do much with)
for i,j in enumerate(z): # x stuff
    for k,l in enumerate(z): # y stuff
        x = xvals
        y = yvals
        z[i][k] = (4500*x[i]**2+15000*x[i]+2000*y[k]**2+y[k]*(7000*x[i]+35000))/1000 # divide by 1000 to convert units from Pa to kPa
        zv[i][k] = ((3*x[i]+5)**2+(-2*y[k]+7*x[i])**2)**(1/2) # To find velocity, find hypotenuse of the x/y velocity components
for m,n in enumerate(zx): # x stuff
    for o,p in enumerate(zx): # y stuff
        x = xvalsc
        y = yvalsc
        zx[m][o] = 4500*x[m]**2+15000*x[m]
        zy[m][o] = 2000*y[o]**2+y[o]*(7000*x[m]+35000)
        zvx[m][o] = 3*x[m]+5
        zvy[m][o] = -2*y[o]+7*x[m]     
####

####
## EXAMPLE 2
## Actually substitute x/y position values for the velocity and pressure fields found above 
## (must manually input pressure field b/c the system has it in terms of variables x, y above that you can't do much with)
#for i,j in enumerate(z): # x stuff
#    for k,l in enumerate(z): # y stuff
#        x = xvals
#        y = yvals
#        z[i][k] = (x[i]**4*(-4500*y[k]**2-3000)-3000*x[i]**3*y[k]-12000*x[i]*y[k]+y[k]**2*(-6000*x[i]**2-4500))/1000 # divide by 1000 to convert units from Pa to kPa
#        zv[i][k] = ((3*x[i]**2*y[k])**2+(4*x[i]+3*y[k])**2)**(1/2) # To find velocity, find hypotenuse of the x/y velocity components
#for m,n in enumerate(zx): # x stuff
#    for o,p in enumerate(zx): # y stuff
#        x = xvalsc
#        y = yvalsc
#        zx[m][o] = x[m]**4*(-4500*y[o]**2 - 3000) - 3000*x[m]**3*y[o]
#        zy[m][o] = -12000*x[m]*y[o] + y[o]**2*(-6000*x[m]**2 - 4500)
#        zvx[m][o] = 3*x[m]**2*y[o]
#        zvy[m][o] = 4*x[m]+3*y[o]
####
        
# plot the pressure and velocity field       
fig = plt.figure(dpi=100)

# shows magnitude of the velocity by color
plt.subplot(221)
plt.pcolormesh(xvals,yvals,zv)
plt.colorbar()
plt.title("Velocity Field (m/s)")

# shows directional velocity by a vector and magnitude by length
plt.subplot(222)
plt.quiver(xvalsc,yvalsc,zvx,zvy)
plt.title("Quiver Plot of Velocity")
plt.show()

# shows magnitude of the pressure by color
plt.subplot(223)
plt.pcolormesh(xvals,yvals,z)
plt.colorbar()
plt.title("Pressure Field (kPa)")

# shows directional pressure by a vector and magnitude by length
plt.subplot(224)
plt.quiver(xvalsc,yvalsc,zx,zy)
plt.title("Quiver Plot of Pressure")

plt.subplots_adjust(hspace=.345,wspace=.225,left=0.1,right=0.9,top=0.920,bottom=0.080)
plt.show()


#"""
#This portion directly uses the Navier-Stokes equation to describe
#the pressure gradient at any location within a fluid. A color mesh
#plot was chosen so that the intensity of the pressure at any location
#is depicted by the color. 
#"""
#
#from sympy.physics.vector import *
#
#x = np.arange(-5,5,0.2)
#y = np.arange(-5,5,0.2)
#
#X, Y = np.meshgrid(x, y)
#v =  X*Y+2*X**2
#vx, vy = np.gradient(v)
#
#def pressureGradient(v,vx,vy):
#    rho = 1000
#    P = []
#    for i,j in enumerate(v):
#        P.append(-rho*v[i]*np.linalg.norm([vx[i],vy[i]]))
#    return P
#
#P = pressureGradient(v,vx,vy)
#
#print(P)
## Create plot of the pressure gradient
#fig = plt.figure(dpi=100)
#plt.pcolormesh(x, y, P)
#plt.colorbar()
#plt.show()

#'''
#This portion directly uses the Navier-Stokes equation to describe
#the pressure gradient at any location within a fluid. A color mesh
#plot was chosen so that the intensity of the pressure at any location
#is depicted by the color. 
#'''
#num = 50
#min_num = -2
#max_num = 2
#dnum = (max_num-min_num)/(num)
#
## Create a grid from -2 to 2 with 50 values
#x = np.array([min_num+i*dnum for i in range(num)])
## Create a copy of the x grid because a square is being represented
#y = x.copy()
#x, y = np.meshgrid(x,y) #, indexing = 'ij', sparse = False
#
## Different veloctity vector functions that can be fed in
#vx = np.cos(x+2*y)
#vy = np.sin(x-2*y)
##vx = np.cos(x)
##vy = np.sin(y)
##vx = -2*x*y
##vy = 3*y**2
#
#
## This forms the full vector
#v = [vx, vy]
#
## Find the magnitude of velocity; the x and y directions were fed in
#vlist = []
#for i in range(num):
#    vlist.append(np.linalg.norm([v[0][i],v[1][i]]))
#
#def divergence(v):
#    gradientlist = []
##    print(v)
#    for i, j in enumerate(v):
##        print(i)
#        gradientlist.append(np.gradient(v[i],axis=i))
#    return np.ufunc.reduce(np.add,gradientlist)
#
#def pressureGradient(v,d):
#    rho = 1000
#    P = []
#    for i,j in enumerate(d):
#        P.append(-rho*v[i]*d[i])
#    return P
#
#
## Find the divergence (velocity dot del) and save as variable d
#d = divergence(v)
#
## Find the pressure gradient, which by the conservation of momentum equation is the velocity*divergence*density
#P = pressureGradient(vlist,d)
#
## Create plot of the pressure gradient
#fig = plt.figure(dpi=100)
#plt.pcolormesh(x, y, P)
#plt.colorbar()
#plt.show()

#'''
#More Pressure Gradient Stuff
#'''
#from sympy.physics.vector import ReferenceFrame
#from sympy.physics.vector import divergence
#import numpy as np
#import matplotlib.pyplot as plt
#
#num = 3
#min_num = -2
#max_num = 2
#dnum = (max_num-min_num)/(num)
#
## Create a grid from -2 to 2 with 50 values
#x = np.array([min_num+i*dnum for i in range(num)])
## Create a copy of the x grid because a square is being represented
#y = x.copy()
#
## Different velocity vector functions that can be fed in
##vx = np.cos(x+2*y)
##vy = np.sin(x-2*y)
##vx = np.cos(x)
##vy = np.sin(y)
#
## Use this to determine what the divergence function will be
#R = ReferenceFrame('R')
#
#v = (3*R[0]+5)*R.x + (7*R[0]+-2*R[1])*R.y
#
#d = divergence(v, R)  
#print(d)
#
## Manually input that divergence function here, and calculate the pressure gradient
#def P(x,y):
#    rho = 1000
#    div = 36*x-6*y+25
#    return rho*div
#
#A = np.fromfunction(lambda x, y: 1000*(36*x-6*y+25),(1000,1000),dtype=float)
#A_div = np.sum(np.gradient(A),axis=0)
#
#plt.figure(dpi=100)
#plt.imshow(A)
