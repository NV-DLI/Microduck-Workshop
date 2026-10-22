"""Explicit surveyed map prior; never obtains simulation state."""
import math
OBSTACLE={'id':'blocking_box01','center_xy_m':[-.31,.025],'size_xy_m':[.04,.05],'yaw_rad':0.,'height_m':.08}
def prior(layout='empty'):
 if layout not in ('empty','one-obstacle'):raise ValueError('Unknown layout')
 return {'kind':'beta_surveyed_course_field','layout':layout,'bounds_xy_m':[-1.,1.,-.5,.5],
  'goal_posts_map_xy_m':[[.93,.1325],[.93,-.1325]],'empty_field_assumed':layout=='empty',
  'obstacles':[] if layout=='empty' else [dict(OBSTACLE)],'information_source':'explicit surveyed static prior; not a sensor observation'}
def bounds(obstacle=OBSTACLE):
 c=obstacle['center_xy_m'];s=obstacle['size_xy_m'];return[c[0]-s[0]/2,c[0]+s[0]/2,c[1]-s[1]/2,c[1]+s[1]/2]
