import numpy as np
import math

"""
structures.py
---------- Functions for generating Structures -------------------
Each function generating a structure should return four arrays. The first two arrays describe the locations of the
exposure points, while the second two describe the locations of the dose-check points. The first of each pair contains
the x-points and the second the y-points. Example:
> (array([-0.5,  0.5,  0.5, -0.5]), array([ 0.5,  0.5, -0.5, -0.5]), array([-1.,  1.,  1., -1.]), array([ 1.,  1., -1., -1.]))
would describe a square of four exposure points around (0,0) with size 1, surrounded by a square of dose-check points
with size 2. The naming convention here is
> (x, y, cx, cy)
and functions returning such a structure should be named 'get_[structure name]'.
"""

from typing import List, Tuple

# Define type aliases. See https://docs.python.org/3/library/typing.html
BasicShape = Tuple[np.ndarray, np.ndarray]
Structure = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Structures = List[Structure]


####### Helper functions

def remove_duplicates(struct: Structure) -> Structure:
    """
    Removes points, if they overlap (only when both points are within both either exposure or dose-check points).
    :param struct: Structure with possible duplicate points
    :return: Cleaned Structure
    """
    assert is_valid_structure(struct)
    x, y = remove_basic_duplicates(struct[0], struct[1])
    cx, cy = remove_basic_duplicates(struct[2], struct[3])
    return x, y, cx, cy


def remove_basic_duplicates(x, y) -> BasicShape:
    """
    Removes a point with index i, if x[i] == y[j] and x[j] == y[j].
    :param x: List of x coordinates
    :param y: List of y coordinates
    :return: Basic shape ([x1,x2,x3,...], [y1,y2,y3,...])
    """
    assert len(x) == len(y)
    nx, ny = [], []
    for i, xi in enumerate(x):
        dupl = False
        for j in np.arange(i + 1, len(x)):
            dupl = dupl or (x[i] == x[j] and y[i] == y[j])
        if not dupl:
            nx.append(xi)
            ny.append(y[i])
    return np.array(nx, dtype=float), np.array(ny, dtype=float)


def is_valid_structure(struct: Structure) -> bool:
    """
    Could the provided structure be valid?
    :param struct: Possibly invalid structure
    :return: False, if structure invalid; True if possibly valid
    """
    return len(struct) == 4 and len(struct[0]) == len(struct[1]) and len(struct[2]) == len(struct[3])


def rot(alpha: float):
    """
    Makes a rotation matrix.
    :param alpha: Rotation angle
    :return: Rotation matrix
    """
    return np.matrix([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])


def translate(dx: float, dy: float, struct: Structure) -> Structure:
    """
    Translates a structure.
    :param dx: Movement in x-direction
    :param dy: Movement in y-direction
    :param struct: Structure array (length=4!)
    :return: Translated structure array
    """
    assert is_valid_structure(struct), 'Provided structure is not valid!'

    return struct[0] + dx, struct[1] + dy, struct[2] + dx, struct[3] + dy


def merge_basic(x0, y0, x1, y1) -> BasicShape:
    """
    Merges several BasicShapes into a single one.
    :return: Merged BasicShape
    """
    return np.hstack([x0, x1]), np.hstack([y0, y1])


def merge(structs: Structures) -> Structure:
    """
    Merges several structures into a single one (keeping exposure and dose-check points seperate).
    :param structs: List of structure arrays (each length=4!)
    :return: Merged Structure
    """
    structs = np.array(structs)
    return np.hstack(structs[:, 0]), np.hstack(structs[:, 1]), np.hstack(structs[:, 2]), np.hstack(structs[:, 3])


####### Basic shapes

def get_basic_circle(r: float, n: int) -> BasicShape:
    """
    Create a list of points for a basic circle.
    :param r: Radius of the circle
    :param n: Number of point used to approximate the circle
    :return: Basic shape ([x1,x2,x3,...], [y1,y2,y3,...])
    """
    v = np.array([r, 0])
    x, y = [], []
    for i in range(n):
        x2, y2 = (v * rot(2 * np.pi / n * i)).A1
        x = np.hstack((x, x2))
        y = np.hstack((y, y2))
    return x, y


def get_basic_square(size: float, n: int) -> BasicShape:
    """
    Create a list of points for a basic square. For more control, use 'get_basic_rectangle'
    First point is in the left upper corner, going in the clockwise direction.
    :param size: length of the rectangle (in x- and y-direction)
    :param n: number of points (in x- and y-direction)
    :return: Basic shape ([x1,x2,x3,...], [y1,y2,y3,...])
    """
    return get_basic_rectangle(width=size, height=size, n_height=n, n_width=n)


def get_basic_rectangle(width: float, height: float, n_width: int, n_height: int) -> BasicShape:
    """
    Create a list of points for a basic rectangle.
    First point is in the left upper corner, going in the clockwise direction.
    :param width: length of the rectangle in x-direction
    :param height: length of the rectangle in y-direction
    :param n_width: number of points in the x-direction
    :param n_height: number of points in the y-direction
    :return: Basic shape ([x1,x2,x3,...], [y1,y2,y3,...])
    """

    n = 2 * n_width + 2 * n_height - 4  # Corner points are shared
    x = np.zeros(n) - 10
    y = np.zeros(n) - 10

    side1 = [0, n_width]
    side2 = [side1[1] - 1, n_width + n_height - 1]
    side3 = [side2[1] - 1, 2 * n_width + n_height - 2]
    side4 = [side3[1] - 1, 2 * n_width + 2 * n_height - 3]

    y[side1[0]:side1[1]] = +height / 2
    y[side3[0]:side3[1]] = -height / 2
    x[side2[0]:side2[1]] = +width / 2
    x[side4[0]:side4[1]] = -width / 2

    for i in range(n_width - 1):
        x[side1[0] + i] = +(i - (n_width - 1) / 2) * (width / (n_width - 1))
        x[side3[0] + i] = -(i - (n_width - 1) / 2) * (width / (n_width - 1))
    for i in range(n_height - 1):
        y[side2[0] + i] = -(i - (n_height - 1) / 2) * (height / (n_height - 1))
        y[side4[0] + i] = +(i - (n_height - 1) / 2) * (height / (n_height - 1))

    return x, y


####### Simple structure elements
def get_square(size: float, n: int, centre_dot: bool = False, dose_check_radius: float = 3,
               corner_comp=False) -> Structure:
    return get_rectangle(width=size, height=size, n_height=n, n_width=n, centre_dot=centre_dot,
                         dose_check_radius=dose_check_radius, corner_comp=corner_comp)


def get_rectangle(width: float, height: float, n_width: int, n_height: int, centre_dot: bool = False,
                  dose_check_radius: float = 3, corner_comp=False) -> Structure:
    """
    Creates a rectangle structure.
    :param width: Width of the resulting rectangle
    :param height: Height of the resulting rectangle
    :param n_width: Number of exposure (and dose-check) points along the x-direction
    :param n_height: Number of exposure (and dose-check) points along the y-direction
    :param centre_dot: Add another exposure point to the middle?
    :param dose_check_radius: Distance of the exposure to dose-check points
    :return: Structure
    """
    assert dose_check_radius > 0
    assert width > dose_check_radius
    assert height > dose_check_radius

    x, y = get_basic_rectangle(width=width - 2 * dose_check_radius, height=height - 2 * dose_check_radius,
                               n_width=n_width, n_height=n_height)
    if corner_comp:
        if type(corner_comp) is bool:
            corner_comp = dose_check_radius
        x1, y1 = get_basic_rectangle(width=width - 2 * corner_comp, height=height - 2 * corner_comp, n_width=2,
                                     n_height=2)
        x, y = merge_basic(x, y, x1, y1)
    if centre_dot:
        x, y = merge_basic(x, y, 0, 0)

    cx, cy = get_basic_rectangle(width=width, height=height, n_width=n_width, n_height=n_height)

    return x, y, cx, cy


def get_square_marker_1(size: float = 100, n: int = 5, centre_dot: bool = True,
                        dose_check_radius: float = 10, corner_comp=False, offset: float = 0) -> Structure:
    sq = get_square(size=size, n=n, centre_dot=centre_dot, dose_check_radius=dose_check_radius, corner_comp=corner_comp)
    squares = merge(
        [translate(dx=-size / 2 - offset, dy=+size / 2 + offset, struct=sq),
         translate(dx=+size / 2 + offset, dy=-size / 2 - offset, struct=sq), ])  # ([], [], [0], [0])])
    return remove_duplicates(squares)


def get_line(length: float, width: float, n: int) -> Structure:
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

    cx = np.concatenate((cx, cx2))
    cy = np.concatenate((cy, cy2))

    return x, y, cx, cy


def get_circle(r: float, n: int = 12, inner_circle: bool = False, centre_dot: bool = False,
               dose_check_radius: float = 3) -> Structure:
    v = np.array([r - dose_check_radius, 0])

    x = np.zeros(0)
    y = np.zeros(0)
    for i in range(n):
        x2, y2 = (v * rot(2 * np.pi / n * i)).A1
        x = np.hstack((x, x2))
        y = np.hstack((y, y2))

    if inner_circle:
        v = np.array([(r - dose_check_radius) / 2, 0])
        n = int(n / 2)
        for i in range(n):
            x2, y2 = (v * rot(2 * np.pi / n * i + 2 * np.pi / (2 * n))).A1
            x = np.hstack((x, x2))
            y = np.hstack((y, y2))

    if centre_dot:
        x = np.hstack((x, 0))
        y = np.hstack((y, 0))

    v = np.array([r, 0])
    cx = np.zeros(0)
    cy = np.zeros(0)
    for i in range(n):
        x2, y2 = (v * rot(2 * np.pi / n * i)).A1
        cx = np.hstack((cx, x2))
        cy = np.hstack((cy, y2))

    return x, y, cx, cy


def get_hexamer(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
                dose_check_radius: float = 3) -> Structure:
    x = np.zeros(0)
    y = np.zeros(0)
    cx = np.zeros(0)
    cy = np.zeros(0)
    v = np.array([0.5 * (dist + 2 * r) / np.sin(np.pi / 6), 0])
    m = 6
    for i in range(m):
        x2, y2 = (v * rot(2 * np.pi / m * i)).A1
        x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot)

        x = np.hstack((x, x1 + x2))
        y = np.hstack((y, y1 + y2))
        cx = np.hstack((cx, cx1 + x2))
        cy = np.hstack((cy, cy1 + y2))

    return x, y, cx, cy


####### Bigger structures

def get_trimer(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
               dose_check_radius: float = 3) -> Structure:
    x = np.zeros(0)
    y = np.zeros(0)
    cx = np.zeros(0)
    cy = np.zeros(0)
    v = np.array([0, 0.5 * (dist + 2 * r) / np.sin(np.pi / 3)])
    m = 3
    for i in range(m):
        x2, y2 = (v * rot(2 * np.pi / m * i)).A1
        x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)

        x = np.hstack((x, x1 + x2))
        y = np.hstack((y, y1 + y2))
        cx = np.hstack((cx, cx1 + x2))
        cy = np.hstack((cy, cy1 + y2))

    return x, y, cx, cy


def get_dimer(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
              dose_check_radius: float = 3) -> Structure:
    x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    x2, y2, cx2, cy2 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    x1 -= (r + dist / 2)
    x2 += (r + dist / 2)
    cx1 -= (r + dist / 2)
    cx2 += (r + dist / 2)

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    cx = np.concatenate((cx1, cx2))
    cy = np.concatenate((cy1, cy2))

    return x, y, cx, cy


def get_asymdimer(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
                  dose_check_radius: float = 3) -> Structure:
    x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    r2 = 1.5 * r
    x2, y2, cx2, cy2 = get_circle(r2, n, inner_circle, centre_dot, dose_check_radius)
    x1 -= r + dist / 2
    x2 += r2 + dist / 2
    cx1 -= r + dist / 2
    cx2 += r2 + dist / 2

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    cx = np.concatenate((cx1, cx2))
    cy = np.concatenate((cy1, cy2))
    return x, y, cx, cy


def get_asymtrimer(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
                   dose_check_radius: float = 3) -> Structure:
    x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    r2 = 1.5 * r
    x2, y2, cx2, cy2 = get_circle(r2, n, inner_circle, centre_dot, dose_check_radius)
    x1 += r + r2 + dist
    cx1 += r + r2 + dist
    # x2 += r2+dist/2

    r3 = 1.5 * r2
    x3, y3, cx3, cy3 = get_circle(r3, n, inner_circle, centre_dot, dose_check_radius)
    x3 -= r2 + r3 + dist
    cx3 -= r2 + r3 + dist

    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))
    cx = np.concatenate((cx1, cx2, cx3))
    cy = np.concatenate((cy1, cy2, cy3))

    return x, y, cx, cy


def get_single(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
               dose_check_radius: float = 3) -> Structure:
    x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)

    # if r >= 50:
    #    x1,y1 = get_circle(r,n=48,inner_circle=True,centre_dot=True)
    # else:
    #    x1, y1 = get_circle(r, n=32, inner_circle=False, centre_dot=True)

    x = x1
    y = y1
    cx = cx1
    cy = cy1

    return x, y, cx, cy


def get_triple(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
               dose_check_radius: float = 3) -> Structure:
    x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    x2, y2, cx2, cy2 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    x3, y3, cx3, cy3 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    x1 -= 2 * r + dist
    x2 += 2 * r + dist
    cx1 -= 2 * r + dist
    cx2 += 2 * r + dist

    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))
    cx = np.concatenate((cx1, cx2, cx3))
    cy = np.concatenate((cy1, cy2, cy3))

    return x, y, cx, cy


def get_triple_rotated(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
                       dose_check_radius: float = 3, alpha: float = 0) -> Structure:
    x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    x2, y2, cx2, cy2 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    x3, y3, cx3, cy3 = get_circle(r, n, inner_circle, centre_dot, dose_check_radius)
    x1 -= 2 * r + dist
    cx1 -= 2 * r + dist

    # v = np.array([r - dose_check_radius, 0])
    v = np.array([2 * r + dist, 0])
    x_rot, y_rot = (v * rot(alpha)).A1

    x2 += x_rot
    cx2 += x_rot
    y2 -= y_rot
    cy2 -= y_rot

    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))
    cx = np.concatenate((cx1, cx2, cx3))
    cy = np.concatenate((cy1, cy2, cy3))

    return x, y, cx, cy


def get_triple00(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
                 dose_check_radius: float = 3) -> Structure:
    return get_triple_rotated(dist, r, n, inner_circle, centre_dot, dose_check_radius, 0)


def get_triple30(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
                 dose_check_radius: float = 3) -> Structure:
    return get_triple_rotated(dist, r, n, inner_circle, centre_dot, dose_check_radius, 2 * np.pi / 12)


def get_triple60(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
                 dose_check_radius: float = 3) -> Structure:
    return get_triple_rotated(dist, r, n, inner_circle, centre_dot, dose_check_radius, 2 * np.pi / 6)


def get_triple90(dist: float, r: float, n: int, inner_circle: bool = False, centre_dot: bool = False,
                 dose_check_radius: float = 3) -> Structure:
    return get_triple_rotated(dist, r, n, inner_circle, centre_dot, dose_check_radius, 2 * np.pi / 4)
