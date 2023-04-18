__author__ = "Ming Song, Yang Liu and Peter A. Kner"
__copyright__ = "Copyright 2023, SISO-SPIM Project"
__credits__ = ["Ming Song", "Yang Liu", "Peter A. Kner"]
__license__ = "GPL"
__version__ = "1.1.0"
__maintainer__ = "Ming Song"
__email__ = "ming.song@uga.edu"
__status__ = "Production"



import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as st
from scipy.special import jn
from tqdm import tqdm
from PIL import Image
from skimage import io
import struct
import os



def crop_center_with_shift(image, crop_size, shift=(0,0)):
    """
    Crop an image from the center, with an optional shift.
    
    Parameters:
        image (numpy.ndarray): The input image with dimensions (N, N).
        crop_size (tuple): A tuple of two integers (W, L) specifying the size of the output image.
        shift (tuple): A tuple of two integers (x, y) specifying the pixel shift from the center.
    
    Returns:
        numpy.ndarray: The cropped image with dimensions (W, L).
    """
    N, _ = image.shape
    W, L = crop_size
    x_shift, y_shift = shift
    x_start = (N - W) // 2 + np.around(x_shift).astype(int)
    y_start = (N - L) // 2 + np.around(y_shift).astype(int)
    x_end = x_start + W
    y_end = y_start + L
    return image[x_start:x_end, y_start:y_end]


def pad_image(image, row, col):
    """
    Pads the input image with zeros to a size (row, col) while keeping the original image centered.

    Parameters:
    image (numpy array): Input image.
    row (int): Desired padded image row size.
    col (int): Desired padded image column size.

    Returns:
    numpy array: Padded image.
    """
    
    # get the original image size
    N = image.shape[0]
    
    # set the center of the image to 1
    center = (N//2, N//2)
    image[center[0], center[1]] = 1

    # calculate the amount of padding required
    pad_row = row - N
    pad_col = col - N

    # calculate the starting row and column indices to insert the original image in the center of the new array
    start_row = pad_row // 2
    start_col = pad_col // 2

    # create a new array of size (row, col) and set all elements to 0
    padded_image = np.zeros((row, col))

    # insert the original image into the center of the new array
    padded_image[start_row:start_row+N, start_col:start_col+N] = image
    
    return padded_image

def get_dots_img(rows, cols, Npix, bessel_dot, bd_range=20):
    """
     1. Generate Dots (1 pixel) image by given size and hex pattern.
     2. Accumulate give bessel beam to the location of generated dots.


    Parameters:
    image (numpy array): Input image.
    row (int): Desired image row size.
    col (int): Desired image column size.
    Npix: Desired distance of 1 pixel dots.
    bessel_dot: Designed single bessel beam.
    bd_range: Desired blank distance at the edge of final image.


    Returns:
    numpy array: Generated final image with hex pattern of bessel beams.
    """
    img = np.zeros((rows,cols))

    img[0,::Npix] = 255
    dot_temp_1d = img[0,:].copy()
    
    x_move_step = np.round(Npix/2).astype(int)
    y_move_step = np.round(Npix*np.sqrt(3)).astype(int)

    bd_w = bessel_dot.shape[0]
    bd_center = int(bd_w/2)

    img[np.arange(0,rows,(y_move_step)),:] = dot_temp_1d.copy()

    img[np.arange(int(np.around(y_move_step/2)),rows,(y_move_step)),:] = np.roll(dot_temp_1d, x_move_step) #+ img[y_move_step,:]

    locs = np.asarray(np.where(img==255)).T
    
    img_bd = np.zeros_like(img)
    for i,j in tqdm(locs):
        # st()
        if i<bd_range or j <bd_range or abs(i-rows)<bd_range or abs(j-cols)<bd_range:continue
        cur_bd = np.roll(bessel_dot, i-bd_center, axis=1)
        cur_bd = np.roll(cur_bd, j-bd_center, axis=0)

        img_bd += cur_bd

    return img_bd 

def save_phases_1bit(img, Npix, phase, alpha, save_path, slm_w, slm_l):
    img[img<=np.max(img)*alpha]=0
    img[img!=0]=255

    for i in range(phase):
        ## angle 1
        cur_angle1_phase = crop_center_with_shift(image=img, crop_size=(slm_w, slm_l), shift=(0,-i*((Npix//phase))))
        temp = cur_angle1_phase.astype(np.uint8)
        final = Image.fromarray(temp)
        final.save(os.path.join(save_path,f'angle_000_phase{i}.bmp'))

        ## angle 2
        cur_angle2_phase = crop_center_with_shift(image=img, crop_size=(slm_w, slm_l), shift=(-i*((Npix//phase//2*np.sqrt(3))),-i*((Npix//phase//2))))
        temp = cur_angle2_phase.astype(np.uint8)
        final = Image.fromarray(temp)
        final.save(os.path.join(save_path,f'angle_060_phase{i}.bmp'))

        ## angle 3
        cur_angle2_phase = crop_center_with_shift(image=img, crop_size=(slm_w, slm_l), shift=(i*((Npix//phase//2*np.sqrt(3))),-i*((Npix//phase//2))))
        temp = cur_angle2_phase.astype(np.uint8)
        final = Image.fromarray(temp)
        final.save(os.path.join(save_path,f'angle_120_phase{i}.bmp'))

    # plt.show()
    return img

def main():
    # rows = 4096
    # cols = 4096

    Npix = 6
    save_path = f"{Npix}pixel_lattice_1bit"
    # Check whether the specified path exists or not
    isExist = os.path.exists(save_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_path)
        print(f"{save_path} directory is created!")

    

    # slm_w = 1536
    # slm_l = 2048



    # rows = 512
    # cols = 512
    slm_w = 100
    slm_l = 200

    wl = 0.488
    nx = 64 ## decide size of bessel beam
    # assert rows>(slm_w+3*Npix+nx/4)
    # assert cols>(slm_l+3*Npix+nx/4)
    # st()
    rows = cols = int(np.max([slm_w+3*Npix+nx/4,slm_l+3*Npix+nx/4])+Npix)

    # st()



    na1 = 0.425 ## from the mask calculation
    na2 = 0.457
    na_p = (na1 + na2) / 2
    dx = wl/na2/2/8
    dp = 1/(nx*dx)
    
    p = dp*np.arange((-nx/2),(nx/2))
    kx,ky = np.meshgrid(p,p,sparse=True)
    rho = np.sqrt(kx**2+ky**2)
    

    xmax = 7.0156
    bconf = dx*xmax
    k = 2*np.pi/wl
    e1 = jn(0,k*rho*na_p)*np.exp(-0.5*(rho/bconf)**2)

    e1_padded = pad_image(e1, rows, cols)






    img = get_dots_img(rows, cols, Npix=Npix, bessel_dot=e1_padded, bd_range=nx/4)
    img_bd_bpp = np.abs(np.fft.fftshift(np.fft.fft2(img)))**2
    plt.figure()
    plt.title('Pattern')
    plt.imshow(img)

    plt.figure()
    plt.title('BPP')
    plt.imshow(img_bd_bpp)
    

    # io.imsave('bd_imgs.tif', img)
    img_bd_bpp = Image.fromarray(img_bd_bpp)
    img_bd_bpp.save('pattern_bpp.tif')

    # st()
    
    temp = save_phases_1bit(img, Npix=Npix, phase=3, alpha=0.3, save_path=save_path, slm_w=slm_w, slm_l=slm_l)

    temp = temp.astype(np.uint8)
    final = Image.fromarray(temp)
    final.save('converted_pattern.bmp')
    im = Image.open('converted_pattern.bmp')
    # im=im.convert('1')
    im.save(f'phase_pattern_px_1bitimg.bmp')

    plt.figure()
    plt.title('Pattern')
    plt.imshow(temp)
    plt.show()


if __name__ == '__main__':
    main()