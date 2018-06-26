
# # Combining Sketch & Tone for Pencil Drawing Production
# ## Python Implementation
# ### Based on the paper "Combining Sketch and Tone for Pencil Drawing Production" by Cewu Lu, Li Xu, Jiaya Jia
# #### International Symposium on Non-Photorealistic Animation and Rendering (NPAR 2012), June 2012
# Project site can be found here:
# http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/pencil_drawing.htm
# 
# Paper PDF - http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/npar12_pencil.pdf
# 
# Draws inspiration from the Matlab implementation by "candtcat1992" - https://github.com/candycat1992/PencilDrawing



# imports
import numpy as np
import cv2
from skimage import io, color, filters, transform, exposure
from scipy import signal, sparse


# Generate Stroke Map
def gen_stroke_map(img, kernel_size, stroke_width=0, num_of_directions=8, smooth_kernel="gauss", gradient_method=0):
    height = img.shape[0] # number of rows, height of the image
    width = img.shape[1] # number of columns, width of the image
    # Let's start with smoothing
    if (smooth_kernel == "gauss"):
        smooth_im = filters.gaussian(img, sigma=np.sqrt(2))
    else:
        smooth_im = filters.median(img) # default is 3x3 kernel size
    # Let's calculate the gradients:
    if not gradient_method:
        # forward gradient: (we pad with zeros)
        imX = np.zeros_like(img)
        diffX = img[: , 1:width] - img[: , 0:width - 1]
        imX[:, 0:width - 1] = diffX
        imY = np.zeros_like(img)
        diffY = img[1:height , :] - img[0:height - 1 , :]
        imY[0:height - 1, :] = diffY
        G = np.sqrt(np.square(imX) + np.square(imY))
    else:
        # Sobel
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        G = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Let's create the basic line segement (horizontal)
    # make sure it is an odd number, so the lines are at the middle
    basic_ker = np.zeros((kernel_size * 2 + 1, kernel_size * 2 + 1))
    basic_ker[kernel_size + 1,:] = 1 # ------- (horizontal line)
    # Let's rotate the lines in the given directions and perform the classification:
    res_map = np.zeros((height, width, num_of_directions))
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        res_map[:,:, d] = signal.convolve2d(G, ker, mode='same')
    max_pixel_indices_map = np.argmax(res_map, axis=2)
    # What does it compute? every direction has a (height X width) matrix. For every pixel in the matrix,
    # np.argmax returns the index of the direction that holds the pixel with the maximum value
    # and thus we get the max_pixel_indices map is a (height X width) matrix with direction numbers.
    # Now we compute the Classification map:
    C = np.zeros_like(res_map)
    for d in range(num_of_directions):
        C[:,:,d] = G * (max_pixel_indices_map == d) # (max_pixel_indices_map == d) is a binary matrix
    # We should now consider the stroke width before we create S'
    if not stroke_width:
        for w in range(1, stroke_width + 1):
            if (kernel_size + 1 - w) >= 0:
                basic_ker[kernel_size + 1 - w, :] = 1
            if (kernel_size + 1 + w) < (kernel_size * 2 + 1):
                basic_ker[kernel_size + 1 + w, :] = 1
    # It's time to compute S':
    S_tag_sep = np.zeros_like(C)
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        S_tag_sep[:,:,d] = signal.convolve2d(C[:,:,d], ker, mode='same')
    S_tag = np.sum(S_tag_sep, axis=2)
    # Remember that S shpuld be an image, thus we need to make sure the values are in [0,1]
    S_tag_normalized = (S_tag - np.min(S_tag.ravel())) / (np.max(S_tag.ravel()) - np.min(S_tag.ravel()))
    # The last step is to invert it (b->w, w->b)
    S = 1 - S_tag_normalized
    return S
    

# Generate Tone Map
# Make sure input image is in [0,1]
def gen_tone_map(img, w_group=0):
    # The first thing we need to do is to calculate the parameters and define weight groups
    w_mat = np.array([[11, 37, 52],
                     [29, 29, 42],
                     [2, 22, 76]])
    w = w_mat[w_group,:]
    # We can now define tone levels like:
    # dark: [0-85]
    # mild: [86-170]
    # bright: [171-255]
    # Assign each pixel a tone level, make 3 lists where each list holds the pixels (indices) of every tone.
    # Use these lists to calculate the parameters for each image.
    
    # For simplicity, we will use the parameters from the paper:
    # for the mild layer:
    u_b = 225
    u_a = 105
    # for the bright layer:
    sigma_b = 9
    # for the dark layer:
    mu_d = 90
    sigma_d = 11
    
    # Let's calculate the new histogram (p(v)):
    num_pixel_vals = 256
    p = np.zeros(num_pixel_vals)
    for v in range(num_pixel_vals):
        p1 = (1 / sigma_b) * np.exp(-(255 - v) / sigma_b)
        if (u_a <= v <= u_b):
            p2 = 1 / (u_b - u_a)
        else:
            p2 = 0
        p3 = (1 / np.sqrt(2 * np.pi * sigma_d)) * np.exp( (-np.square(v - mu_d)) / (2 * np.square(sigma_d)) )
        p[v] = w[0] * p1 + w[1] * p2 + w[2] * p3 * 0.01
    # normalize the histogram:
    p_normalized = p / np.sum(p)
    # calculate the CDF of the desired histogram:
    P = np.cumsum(p_normalized)
    # calculate the original histogram:
    h = exposure.histogram(img, nbins=256)
    # CDF of original:
    H = np.cumsum(h / np.sum(h))
    # histogram matching:
    lut = np.zeros_like(p)
    for v in range(num_pixel_vals):
        # find the closest value:
        dist = np.abs(P - H[v])
        argmin_dist = np.argmin(dist)
        lut[v] = argmin_dist
    lut_normalized = lut / num_pixel_vals
    J = lut_normalized[(255 * img).astype(np.int)]
    # smooth:
    J_smoothed = filters.gaussian(J, sigma=np.sqrt(2))
    return J_smoothed
    


# Generate Pencil Texture:
def gen_pencil_texture(img, H, J):
    # define the regularization parameter:
    lamda = 0.2
    height = img.shape[0]
    width = img.shape[1]
    # Adjust the input to correspond
#     H_res = transform.resize(H,(height, width))
    H_res = cv2.resize(H, (width, height), interpolation=cv2.INTER_CUBIC)
    H_res_reshaped = np.reshape(H_res, (height * width, 1))
    logH = np.log(H_res_reshaped)
    
#     J_res = transform.resize(J,(height, width))
    J_res = cv2.resize(J, (width, height), interpolation=cv2.INTER_CUBIC)
    J_res_reshaped = np.reshape(J_res, (height * width, 1))
    logJ = np.log(J_res_reshaped)
    
    # In order to use Conjugate Gradient method we need to prepare some sparse matrices:
    logH_sparse = sparse.spdiags(logH.ravel(), 0, height*width, height*width) # 0 - from main diagonal
    e = np.ones((height * width, 1))
    ee = np.concatenate((-e,e), axis=1)
    diags_x = [0, height*width]
    diags_y = [0, 1]
    dx = sparse.spdiags(ee.T, diags_x, height*width, height*width)
    dy = sparse.spdiags(ee.T, diags_y, height*width, height*width)
    
    # Compute matrix X and b: (to solve Ax = b)
    A = lamda * ((dx @ dx.T) + (dy @ dy.T)) + logH_sparse.T @ logH_sparse
    b = logH_sparse.T @ logJ
    
    # Conjugate Gradient
    beta = sparse.linalg.cg(A, b, tol=1e-6, maxiter=60)
    
    # Adjust the result
    beta_reshaped = np.reshape(beta[0], (height, width))
    
    # The final pencil texture map T
    T = np.power(H_res, beta_reshaped)
    
    return T
    

# It's time to pack it all up (WOOHOOO!)
def gen_pencil_drawing(img, kernel_size, stroke_width=0, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=0, rgb=False, w_group=0, pencil_texture_path=""):
    if not rgb:
        # Grayscale image:
        im = img
    else:
        # RGB image:
        yuv_img = color.rgb2yuv(img)
        im = yuv_img[:,:,0]
    # Generate the Stroke Map:
    S = gen_stroke_map(im, kernel_size, stroke_width=stroke_width, num_of_directions=num_of_directions,
                       smooth_kernel=smooth_kernel, gradient_method=gradient_method)
    # Generate the Tone Map:
    J = gen_tone_map(im, w_group=w_group)
    
    # Read the pencil texture:
    if not pencil_texture_path:
        pencil_texture = io.imread('./pencils/pencil0.jpg', as_gray=True)
    else:
        pencil_texture = io.imread(pencil_texture_path, as_gray=True)
    # Generate the Pencil Texture Map:
    T = gen_pencil_texture(im, pencil_texture, J)
    
    # The final Y channel:
    R = np.multiply(S, T)
    
    if not rgb:
        return R
    else:
        yuv_img[:,:,0] = R
        return color.yuv2rgb(yuv_img)


