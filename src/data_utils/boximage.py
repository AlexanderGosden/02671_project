from math import ceil, floor
import numpy as np

def boximage(box_width, box_height, width = 128, height = 128, center = None):
    # Create a box image with anti-aliased edges
    # box_width and box_height are the width and height of the box
    # width and height are the dimensions of the image
    # center is the center of the box


    # If center is not provided, place the box in the center of the image
    if center is None:
        center = ((height-1)/2, (width-1)/2)

    im = np.zeros((height, width), dtype=np.uint8)

    y_start = max(center[0]-box_height/2, 0)
    y_end = min(center[0]+box_height/2, height-1)

    x_start = max(center[1]-box_width/2, 0)
    x_end = min(center[1]+box_width/2, width-1)

    # Anti-alias the edges
    im[floor(y_start), ceil(x_start):floor(x_end+1)] = 255*(ceil(y_start)-y_start)
    im[ceil(y_end), ceil(x_start):floor(x_end+1)] = 255*(y_end - floor(y_end))
    im[ceil(y_start):floor(y_end+1), floor(x_start)] = 255*(ceil(x_start) - x_start)
    im[ceil(y_start):floor(y_end+1), ceil(x_end)] = 255*(x_end - floor(x_end))

    # Anti-alias the corners
    im[floor(y_start), floor(x_start)] = 255*(ceil(y_start) - y_start)*(ceil(x_start) - x_start)
    im[floor(y_start), ceil(x_end)] = 255*(ceil(y_start) - y_start)*(x_end - floor(x_end))
    im[ceil(y_end), floor(x_start)] = 255*(y_end - floor(y_end))*(ceil(x_start) - x_start)
    im[ceil(y_end), ceil(x_end)] = 255*(y_end - floor(y_end))*(x_end - floor(x_end))

    # Fill the box
    im[ceil(y_start):floor(y_end+1), ceil(x_start):floor(x_end+1)] = 255


    return im
