import numpy as np

#---------- Functions for generating Structures -------------------

def rot(alpha):
    return np.matrix( [[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]] )

def get_circle(r,n=12,inner_circle=False,centre_dot=False,dose_check_radius = 3):
    v = np.array( [r-dose_check_radius,0] )

    x = np.zeros(0)
    y = np.zeros(0)
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )

    if inner_circle:
        v = np.array( [(r-dose_check_radius)/2,0] )
        n = int(n/2)
        for i in range(n):
            x2, y2 = (v*rot(2*np.pi/n*i+2*np.pi/(2*n))).A1
            x = np.hstack( (x,x2) )
            y = np.hstack( (y,y2) )

    if centre_dot:
        x = np.hstack( (x,0) )
        y = np.hstack( (y,0) )


    v = np.array( [r,0] )
    cx = np.zeros(0)
    cy = np.zeros(0)
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        cx = np.hstack( (cx,x2) )
        cy = np.hstack( (cy,y2) )

    return x,y, cx, cy

def get_trimer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x = np.zeros(0)
    y = np.zeros(0)
    cx = np.zeros(0)
    cy = np.zeros(0)
    v = np.array( [0,0.5*(dist+2*r)/np.sin(np.pi/3)] )
    m = 3
    for i in range(m):
        x2, y2 = (v*rot(2*np.pi/m*i)).A1
        x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot, dose_check_radius)

        x = np.hstack( (x,x1+x2) )
        y = np.hstack( (y,y1+y2) )
        cx = np.hstack( (cx,cx1+x2) )
        cy = np.hstack( (cy,cy1+y2) )

    x += 500
    y += 500
    cx += 500
    cy += 500

    return x,y,cx,cy

def get_dimer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x2,y2,cx2,cy2 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x1 -= (r+dist/2)
    x2 += (r+dist/2)
    cx1 -= (r+dist/2)
    cx2 += (r+dist/2)

    x = np.concatenate((x1,x2))+500
    y = np.concatenate((y1,y2))+500
    cx = np.concatenate((cx1,cx2))+500
    cy = np.concatenate((cy1,cy2))+500
    # cy = np.concatenate((y1,y2))+500

    return x,y, cx, cy

def get_hexamer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x = np.zeros(0)
    y = np.zeros(0)
    cx = np.zeros(0)
    cy = np.zeros(0)
    v = np.array( [0.5*(dist+2*r)/np.sin(np.pi/6),0] )
    m = 6
    for i in range(m):
        x2, y2 = (v*rot(2*np.pi/m*i)).A1
        x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot)

        x = np.hstack( (x,x1+x2) )
        y = np.hstack( (y,y1+y2) )
        cx = np.hstack( (cx,cx1+x2) )
        cy = np.hstack( (cy,cy1+y2) )

    x += 500
    y += 500
    cx += 500
    cy += 500

    return x,y,cx,cy


def get_asymdimer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    r2 = 1.5*r
    x2,y2,cx2,cy2 = get_circle(r2,n,inner_circle, centre_dot,dose_check_radius)
    x1 -= r+dist/2
    x2 += r2+dist/2
    cx1 -= r+dist/2
    cx2 += r2+dist/2

    x = np.concatenate((x1,x2))+500
    y = np.concatenate((y1,y2))+500
    cx = np.concatenate((cx1,cx2))+500
    cy = np.concatenate((cy1,cy2))+500
    return x,y,cx,cy


def get_asymtrimer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    r2 = 1.5*r
    x2,y2,cx2,cy2 = get_circle(r2,n,inner_circle, centre_dot,dose_check_radius)
    x1 += r+r2+dist
    cx1 += r+r2+dist
    #x2 += r2+dist/2

    r3 = 1.5*r2
    x3,y3,cx3,cy3 = get_circle(r3,n,inner_circle, centre_dot,dose_check_radius)
    x3 -= r2+r3+dist
    cx3 -= r2+r3+dist

    x = np.concatenate((x1,x2,x3))+500
    y = np.concatenate((y1,y2,y3))+500
    cx = np.concatenate((cx1,cx2,cx3))+500
    cy = np.concatenate((cy1,cy2,cy3))+500

    return x,y,cx,cy


def get_single(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot,dose_check_radius)

    #if r >= 50:
    #    x1,y1 = get_circle(r,n=48,inner_circle=True,centre_dot=True)
    #else:
    #    x1, y1 = get_circle(r, n=32, inner_circle=False, centre_dot=True)

    x = x1+500
    y = y1+500
    cx = cx1+500
    cy = cy1+500

    return x,y,cx,cy



def get_triple(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x2,y2,cx2,cy2 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x3,y3,cx3,cy3 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x1 -= 2*r+dist
    x2 += 2*r+dist
    cx1 -= 2*r+dist
    cx2 += 2*r+dist

    x = np.concatenate((x1,x2,x3))+500
    y = np.concatenate((y1,y2,y3))+500
    cx = np.concatenate((cx1, cx2, cx3)) + 500
    cy = np.concatenate((cy1, cy2, cy3)) + 500

    return x,y,cx,cy



def get_triple_rotated(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3,alpha = 0):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x2,y2,cx2,cy2 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x3,y3,cx3,cy3 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x1 -= 2*r+dist
    cx1 -= 2*r+dist

    #v = np.array([r - dose_check_radius, 0])
    v = np.array([2*r+dist, 0])
    x_rot, y_rot = (v * rot(alpha)).A1

    x2 += x_rot
    cx2 += x_rot
    y2 -= y_rot
    cy2 -= y_rot

    x = np.concatenate((x1,x2,x3))+500
    y = np.concatenate((y1,y2,y3))+500
    cx = np.concatenate((cx1, cx2, cx3)) + 500
    cy = np.concatenate((cy1, cy2, cy3)) + 500

    return x,y,cx,cy

def get_triple00(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    return get_triple_rotated(dist,r,n,inner_circle,centre_dot,dose_check_radius,0)

def get_triple30(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    return get_triple_rotated(dist,r,n,inner_circle,centre_dot,dose_check_radius,2*np.pi/12)

def get_triple60(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    return get_triple_rotated(dist,r,n,inner_circle,centre_dot,dose_check_radius,2*np.pi/6)

def get_triple90(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    return get_triple_rotated(dist,r,n,inner_circle,centre_dot,dose_check_radius,2*np.pi/4)

def get_line(length, width, n):
    dist = length / n
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] += i * dist

    cx = np.zeros(n)
    cy = np.zeros(n) + width / 2
    for i in range(n):
        cx[i] += i * dist

    cx2 = np.zeros(n)
    cy2 = np.zeros(n) - width / 2
    for i in range(n):
        cx2[i] += i * dist

    cx = np.concatenate((cx,cx2))
    cy = np.concatenate((cy, cy2))

    return x,y,cx,cy