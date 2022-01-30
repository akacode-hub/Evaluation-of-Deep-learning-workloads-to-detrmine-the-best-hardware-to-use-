import numpy as np
import random
import matplotlib.pyplot as plt

CONTOUR_LEVELS = np.geomspace(0.0001, 250, 50)

def get_true_pos(center, radius):

    rand_r = random.uniform(0.0, 1.0)
    rand_t = random.uniform(0.0, 1.0)
    r = radius * np.sqrt(rand_r)
    theta = 2 * np.pi * rand_t
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.array([x, y])

def get_equidist_points(center, radius, num_points):

    equ_pts = []
    for i in range(num_points):
        theta = 2.0 * np.pi * i/ num_points
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        equ_pts.append([x, y])
    
    return np.array(equ_pts)

def get_measurements(equ_pts, true_pos):

    measurements = []
    for equ_pt in equ_pts:

        dist = np.linalg.norm(true_pos - equ_pt)
        measurement = dist + np.random.normal(0, sigi)

        while measurement<=0:
            measurement = dist + np.random.normal(0, sigi)

        measurements.append(measurement)

    return np.array(measurements)

def get_MAP_contour(equ_pts, measurements, quad_range, num_grid_pts):

    min_, max_ = quad_range[0], quad_range[1]
    xgrid = np.linspace(min_, max_, num_grid_pts)
    ygrid = np.linspace(min_, max_, num_grid_pts)
    mat = np.array(np.meshgrid(xgrid, ygrid))

    contours = np.zeros((num_grid_pts, num_grid_pts), dtype='float')
    for i in range(num_grid_pts):
        for j in range(num_grid_pts):
            x1 = mat[0][i][j]
            x2 = mat[1][i][j]
            pt = np.array([x1, x2])
            contours[i][j] = get_MAP_obj(pt, equ_pts, measurements, num_pts)
    
    return contours, mat

def plot_equilevel_contours(equ_pts, measurements, quad_range, num_grid_pts):

    contours, grid = get_MAP_contour(equ_pts, measurements, quad_range, num_grid_pts)

    ax = plt.gca()

    unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False)
    ax.add_artist(unit_circle)

    plt.contour(grid[0], grid[1], contours, cmap='cividis_r', levels=CONTOUR_LEVELS)

    for (pt_i, r_i) in zip(equ_pts, measurements):
        
        print('r i ',r_i)
        x, y = pt_i[0], pt_i[1]
        plt.plot((x), (y), 'o', color='g', markerfacecolor='none')
        # range_circle = plt.Circle((x, y), r_i, color='g', fill=False)
        # ax.add_artist(range_circle)

    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_title("MAP estimation objective contours, K = " + str(len(measurements)))

    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.plot([true_pos[0]], [true_pos[1]], '+', color='r')
    plt.colorbar();
    plt.show()
    
def get_MAP_obj(pt, equ_pts, measurements, num_pts): ##(1, 2)

    sigma_mat = np.array([[sigx**2, 0], [0, sigy**2]])

    prior = np.matmul(pt, np.linalg.inv(sigma_mat))
    prior = np.matmul(prior, pt.T)

    measure_sum = 0
    for equ_pt, r_i in zip(equ_pts, measurements):
        d_i = np.linalg.norm(pt - equ_pt)
        measure = (r_i - d_i)**2/sigi**2
        measure_sum += measure

    return prior + measure_sum

if __name__ == "__main__":

    sigx, sigy = 0.25, 0.25
    sigi = 0.3
    num_points = [40, 20, 30, 40]
    center = [0, 0]
    radius = 1
    quad_range = (-2, 2)
    num_grid_pts = 128

    true_pos = get_true_pos(center, radius)

    for num_pts in num_points:
        equ_pts = get_equidist_points(center, radius, num_pts)
        measurements = get_measurements(equ_pts, true_pos)
        print('measurements ',measurements)
        plot_equilevel_contours(equ_pts, measurements, quad_range, num_grid_pts)
