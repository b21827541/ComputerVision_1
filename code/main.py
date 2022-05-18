import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import math
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    #load image
    image = mpimg.imread('./images/Cars56.png')
    #img copy
    img = image.copy()

    #xml file read for plates location
    tree = ET.parse("./annotations/Cars0.xml")
    root = tree.getroot()
    for item in root.iter('annotation'):
        for xmin in item.iter('xmin'):
            x_min = int(xmin.text)
        for ymin in item.iter('ymin'):
            y_min = int(ymin.text)
        for xmax in item.iter('xmax'):
            x_max = int(xmax.text)
        for ymax in item.iter('ymax'):
            y_max = int(ymax.text)

    #gray scale convert
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #noise remove
    image = cv2.GaussianBlur(gray, (3, 3), 0)
    image = (image*255).astype(np.uint8)

    #canny edge detector

    edged_image = cv2.Canny(image, 30, 250)
    plt.imshow(edged_image, 'gray')
    plt.show()

    borderLen = 5                         #The width to zero out the borders, counted in pixels
    lenx, leny = edged_image.shape

    edged_image[0:borderLen, 0:leny] = 0
    edged_image[lenx-borderLen:lenx, 0:leny] = 0
    edged_image[0:lenx, 0:borderLen] = 0
    edged_image[0:lenx, leny-borderLen:leny] = 0

    #image shape
    img_shape = edged_image.shape
    x_max = img_shape[0]
    y_max = img_shape[1]

    #min max values for rho and theta
    r_min = 0.0
    r_max = math.hypot(x_max, y_max)
    r_dim = 200
    theta_dim = 180
    theta_max = 1.0 * math.pi
    thetas = np.deg2rad(np.arange(-90.0, 90.0, 1.0))
    diagonal = int(round(np.sqrt(x_max*x_max + y_max * y_max)))
    rhos = np.linspace(-diagonal, diagonal, diagonal * 2)

    hough_space = np.zeros((r_dim, theta_dim))
    list_lines = []
    #read image values
    for i in range(x_max):
        for j in range(y_max):
            #check white pixels in edged
            if edged_image[i, j] == 255:
                #Hough transform for every theta value
                for idx_theta in range(0, theta_dim, 3):
                    #theta to rad
                    theta = idx_theta * theta_max / theta_dim
                    r = i * math.cos(theta) + j * math.sin(theta)
                    ir = int(r_dim * r / r_max)
                    hough_space[ir, idx_theta] = hough_space[ir, idx_theta] + 1
                    #take values higher than treshold
                    if hough_space[ir, idx_theta] > img_shape[0]/1.6 and idx_theta not in list_lines and ir not in list_lines:
                        list_lines.append([r, theta, j])

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.autoscale(False)

    #draw lines on image
    for line in list_lines:

        r = line[0]
        theta = line[1]

        interval_y_label = [-line[2]-15, line[2]+15]

        px = []
        py = []

        for y in interval_y_label:
            px.append(math.cos(-theta) * y - math.sin(-theta) * r)
            py.append(math.sin(-theta) * y + math.cos(-theta) * r)

        ax.plot(px, py, color='red', linewidth=1)
    plt.show()

    #hough space plot
    plt.imshow(hough_space, origin='lower')
    plt.xlim(0, theta_dim)
    plt.ylim(0, r_dim)

    tick_locs = [i for i in range(0, theta_dim, 40)]
    tick_lbls = [round((1.0 * i * theta_max) / theta_dim, 1) for i in range(0, theta_dim, 40)]
    plt.xticks(tick_locs, tick_lbls)

    tick_locs = [i for i in range(0, r_dim, 20)]
    tick_lbls = [round((1.0 * i * r_max ) / r_dim, 1) for i in range(0, r_dim, 20)]
    plt.yticks(tick_locs, tick_lbls)

    plt.xlabel(r'Theta')
    plt.ylabel(r'r')
    plt.title('Hough Space')
    plt.show()

