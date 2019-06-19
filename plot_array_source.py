# -*- coding: utf-8 -*-
"""
@author: Jacopo Carrani
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl


dirFile = os.path.dirname(os.path.join('D:\\Final Year Project\\Figures',
                          'NicePlotProductivity.py'))
plt.style.use(os.path.join(dirFile, 'FYP_Figures.mplstyle'))

x=[]
y=[]
dist=1
angles = np.radians(np.arange(0,365,5))
arr_centre = np.array([1.5,.5,1.5]).reshape(1,3)
arr = arr_centre[0,-1]*np.ones((3,4))
arr[0,0] = arr_centre[0,0]-0.045*3
arr[0,1] = arr_centre[0,0]-0.015*3
arr[0,2] = arr_centre[0,0]+0.015*3
arr[0,3] = arr_centre[0,0]+0.045*3
arr[1,:] = arr_centre[0,1]

for ang in range(0,37):
    x.append(np.cos(angles[ang]) * dist + arr_centre[0,0])
    y.append(np.sin(angles[ang]) * dist + arr_centre[0,1])
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1)
style="Simple,tail_width=0.5,head_width=4,head_length=8"
kw = dict(arrowstyle=style, color="k")
ax.add_patch(patches.FancyArrowPatch((1.8,0.5), (1.58,0.82),connectionstyle="arc3,rad=0.3", **kw))

#ax.add_patch(matplotlib.patches.Rectangle(xy=(1.4,0.955),width=0.2,height=0.11,fc='w',ec='k'))
ax.text(1.8,0.65,'DOA')
plt.plot([x[0],1.5],[0.5,y[0]],'k',ls='dashed',zorder=0)
plt.plot([1.5,x[15]],[0.5,y[15]],'k',ls='dashed',zorder=0)
plt.plot(x,y,'o',label='Source',markersize=8,mew=2,mfc='none')
plt.plot(x[15],y[15],'ro',label='Active Source',markersize=8,mew=2)

plt.plot(arr[0,:],arr[1,:],'x',label='Microphone',mew=2,markersize=8,color='orange')
plt.legend()
plt.xlim([0,3])
plt.ylim([0,2])
ax.get_yaxis().set_ticks([])
ax.get_xaxis().set_ticks([])

plt.title('Room with Microphone Array-Source Setup')
plt.savefig(os.path.join(dirFile,'roomsourcearray.pdf'),dpi=300)
plt.show()