# Import Library
import cv2     # OpenCV Library to Process the Image
import math #Math Library to Calculating
import numpy as np   # Numpy Library
import matplotlib.pyplot as plt # Matplotlib Library to Plot Image and Diagram

# Input the Image
img = cv2.imread('noise.jpg', 0)
f = cv2.imread('lena_high_contrast.png',0)


# PART I, DENOISE IMAGE

# 1, Create Median Filter
def median_filter(img, filter_size):
    m, n = img.shape 
    t = math.floor((filter_size-1)/2)
    img_new = np.zeros((m, n))
    temp = []
    for i in range(m):
        for j in range(n):
            for z in range(filter_size):
                
                if (i + z - t < 0) or (i + z - t > m -  1): 
                    for h in range(filter_size):
                        temp.append(0)
                else:
                    if (j + z - t < 0) or (j + t > n - 1): 
                        temp.append(0)
                    else:
                        for k in range (filter_size): 
                            temp.append(img[ i + z - t][ j + k - t])

            temp = sorted(temp)
            img_new[i][j] = temp[math.floor((len(temp)+1)/2)]
            temp = []
    return img_new


# 2, Processing the Image

# Create region to Plot Image and Its Histogram
fig = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3, ax4)= fig.subplots(2, 2)

# Origin Image
ax1.imshow(img, cmap='gray')
ax1.set_title("Original Image")

# Histogram of Origin Image
ax3.hist(img.ravel(), bins = 256, range = [0, 255])
ax3.set_title("Histogram of Image")
ax3.set_xlabel('Gray level')
ax3.set_ylabel('Number of Pixel')

# Filtered Image
filtered_Img = median_filter(img, 3) 
ax2.imshow(filtered_Img, cmap='gray')
ax2.set_title("Filtered Image")

# Histogram of Origin Image
ax4.hist(filtered_Img.ravel(), bins = 256, range = [0, 255])
ax4.set_title("Histogram of Filtered Image")
ax4.set_xlabel('Gray level')
ax4.set_ylabel('Number of Pixel')

plt.show()

# PART II, BRIGHNESS AND CONTRAST IMAGE

# 1, Show the Image and its histogram and cdf_normalized

# Show the Image
plt.imshow(f,cmap="gray")
plt.title("Original Image")
plt.axis('off')
plt.show()

# Calculate the Histogram value
hist, bins = np.histogram(f.flatten(), bins=256, range=[0,255])

# Calculate the cdf, N.cdf
cdf = hist.cumsum()
cdf_normalized = cdf / cdf.max()

# Show results
fig, ax = plt.subplots(figsize=(5,5))

# Plot histogram of image
ax.hist(f.flatten(), bins=256, range=[0,255], color='r')
ax.set_xlabel('Intensity')
ax.set_ylabel('Histogram')

# Plot cdf_normalized
ax2 = ax.twinx()
ax2.plot(cdf_normalized, color='b')
ax2.set_ylabel('cdf')
ax2.set_ylim(0,1)

plt.show()

# 2, Equalization image

# Calculate equalization
equ = cv2.equalizeHist(f)

plt.imshow(equ, cmap='gray')
plt.axis('off')
plt.show()

# Calculate histogram equalisation
hist_equ, bins_equ = np.histogram(equ.flatten(),bins=256, range=[0,256])

# Calculate cdf, n.cdf
cdf_equ= hist.cumsum()
cdf_normalized_equ= cdf_equ/ cdf_equ.max()

# Show results
fig, ax = plt.subplots(figsize=(5,5))

ax.hist(equ.flatten(), bins=256, range=[0,256], color='r')
ax.set_xlabel('Intensity')
ax.set_ylabel('Histogram')
ax.set_xlim(0,255)

ax2 = ax.twinx()
ax2.plot(cdf_normalized_equ, color='b')
ax2.set_ylabel('cdf')
ax2.set_ylim(0,1)

plt.show()

# 3, CLAHE

# Apply CLAHE to the image
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahef = clahe.apply(f)

plt.imshow(clahef, cmap='gray')
plt.axis('off')
plt.show()

# Calcualte histogram
hist_clahe, bins_clahe = np.histogram(clahef.flatten(),bins=256, range=[0,256])

# Calculate cdf, n.cdf
cdf_clahe = hist_clahe.cumsum()
cdf_normalized_clahe = cdf_clahe / cdf_clahe.max()

# Show results
fig, ax = plt.subplots(figsize=(5,5))

ax.hist(clahef.flatten(),
        bins=256,
        range=[0,256],
        color='r')
ax.set_xlabel('Intensity')
ax.set_ylabel('Histogram')
ax.set_xlim(0,255)
ax.set_ylim(0,20000)

ax2 = ax.twinx()
ax2.plot(cdf_normalized, color='b')
ax2.set_ylabel('cdf')
ax2.set_ylim(0,1)

plt.show()


# PART III, SHARPENING IMAGE

# 1, Normalized the Image in range [0, 1]
N_img = img/255

# 2, Transform Image into Frequency Domain by using FFT
F = np.fft.fftshift(np.fft.fft2(N_img))

plt.figure(figsize = (15, 8))
plt.imshow(np.log1p(np.abs(F)), cmap='gray')
plt.show()

# 3, Create Laplacian Filter
x, y = F.shape
Laplacian_Filter = np.zeros((x, y), dtype = np.float32)
for u in range(x):
    for v in range(y):
        Laplacian_Filter[u, v] = -4*(np.pi**2)*((u  - x/2)**2 + (v - y/2)**2)

plt.figure(figsize = (15, 8))
plt.imshow(Laplacian_Filter, cmap='gray')
plt.show()

# 4, Laplacian Image
T = Laplacian_Filter * F
Laplacian_Image = np.real(np.fft.ifft2(np.fft.ifftshift(T)))

# 5, Normalize Laplacian Image in range [-1, 1]
origin_Range = np.max(Laplacian_Image) - np.min(Laplacian_Image)
new_Range = 2
N_Laplacian_Image = new_Range*(Laplacian_Image - np.min(Laplacian_Image))/origin_Range -1

plt.figure(figsize = (15, 8))
plt.imshow(N_Laplacian_Image, cmap='gray')
plt.show()

# 6, Image Enhancement
c = -1
g = N_img + c*N_Laplacian_Image
g = np.clip(g, 0, 1) # 0, 1 be the normalize of f(x, y) and g(x, y) must be the same

# 7, Processing the Image

# Create region to Plot Image and Its Histogram
fig = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3, ax4)= fig.subplots(2, 2)

# Show the Origin Image
ax1.imshow(img, cmap='gray')
ax1.set_title("Original Image")

# Histogram of Origin Image
ax3.hist(img.ravel(), bins = 256, range = [0, 255])
ax3.set_title("Histogram of Image")
ax3.set_xlabel('Gray level')
ax3.set_ylabel('Number of Pixel')

# Show the Filtered Image
ax2.imshow(g, cmap='gray')
ax2.set_title("Filtered Image")

# Histogram of Origin Image
ax4.hist(g.ravel(), bins = 256, range = [0, 1])
ax4.set_title("Histogram of Filtered Image")
ax4.set_xlabel('Gray level')
ax4.set_ylabel('Number of Pixel')

plt.show()
