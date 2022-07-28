import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils



def find_max_cluster(data,k,labels,centers):
  # finding the largest cluster in the data
  # input: 
  # data - column vector of the image pixel values 
  # k    - number of clusters
  # labales - the label that given to each cluster (0,1,2..)
  # centers - the value of the center data of each cluster [gray level RGB]
  # output:
  # max_cluster -  the number of pixels in the largest cluster 
  # center_max - the RGB data of the lergest cluster 
  # ind - the index of the largest cluster  
  max_cluster = 0
  for i in range(0,k):
    cluster = data[labels==i]
    if len(cluster) > max_cluster:
      max_cluster = len(cluster)
      center_max = centers[i]
      ind = i 
  
  return max_cluster ,center_max ,ind


def calc_cog_distance_from_bbx_center(segmented_image,center_max, plot=0):
  # find the cog of the cluster in x,y image coordinate system and calc the distance from the bbox center
  # input :
  # segmented_image - the image after the segmentation (one color for each cluster)
  # center_max - the color of the largest segment 
  # plot - if 1 the show the output image
  # output: 
  # cog_x,cog_y - the center of gravity of the largest cluster 
  # bbox_center_x, bbox_center_y - the center of the input image wich is the enter of the detection bbox 
  y,x = np.where( (segmented_image[:,:,0]==center_max[0]) & (segmented_image[:,:,1]==center_max[1]) & (segmented_image[:,:,2]==center_max[2]) )
  cog_y = np.mean(y)
  cog_x = np.mean(x)
  im_size = segmented_image.shape
  bbox_center_y = im_size[0]/2
  bbox_center_x = im_size[1]/2

  distance = ( (bbox_center_x-cog_x)**2 + (bbox_center_y-cog_y)**2 )**0.5
  # Using cv2.putText() method
  segmented_image = cv2.putText(segmented_image, 'X', (int(cog_x),int(cog_y)), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
  segmented_image = cv2.putText(segmented_image, 'o', (int(bbox_center_x),int(bbox_center_y)),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
  if plot:
    plt.imshow(segmented_image)
  return cog_x, cog_y, bbox_center_x, bbox_center_y
  

def calc_balloon_radius(segmented_image,labels,cluster_to_keep,plot=0):
  # finding the contour of the balloon cluster and fit a circle (or an allipse) to it's shape and claculate the radius (or a bounding rectangle)
  # input:  
  # segmented_image - the image after the segmentation (one color for each cluster)
  # labales - the label that given to each cluster (0,1,2..)
  # cluster_to_keep - the index of the ballon cluster
  # plot - if 1 the show the output image
  # output: 
  # radius - balloon radius [pix]
  # circle_center - the center o fthe balloon
  # cog - cog of the found contour 
  #######

  # disable all cluster that is not the balloon (turn the pixel into black)
  masked_image = np.copy(segmented_image)
  # convert to the shape of a vector of pixel values
  masked_image = masked_image.reshape((-1, 3))
  # color (i.e cluster) to keep
  cluster = cluster_to_keep
  masked_image[labels != cluster] = [0, 0, 0]
  masked_image[labels == cluster] = [255, 255, 255]
  # convert back to original shape
  masked_image = masked_image.reshape(segmented_image.shape)
  # find circle contour in image 
  imgray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
  ret, imgray = cv2.threshold(imgray, 127, 255, 0)
  mask = cv2.erode(imgray,None, iterations=8)
  mask = cv2.dilate(imgray,None, iterations=8)
  # find contours in the mask and initialize the current
  # (x, y) center of the ball
  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  # only proceed if at least one contour was found
  if len(cnts) > 0:
  # find the largest contour in the mask, then use
  # it to compute the minimum enclosing circle and
  # centroid
      c = max(cnts, key=cv2.contourArea)
      ((x,y), radius) = cv2.minEnclosingCircle(c)
      circle_center = (x,y)
      M = cv2.moments(c)
      cog = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
      if radius > 10:
          # elps = cv2.fitEllipse(c)
          # cv2.ellipse(segmented_image,elps,(0,255,0),2)
          # draw the circle and centroid on the frame,
          cv2.circle(segmented_image,(int(x), int(y)), int(radius), (0,0,255), 2)
          cv2.circle(segmented_image,(int(x), int(y)), 5, 0, -1)
          cv2.line(segmented_image, (int(x), int(y)), (int(x), int(y)+int(radius)), (0,0,255), 2) 
  if plot:
    plt.imshow(segmented_image)
  
  return radius, circle_center, cog


def find_seg_image(image,num_clusters=4,plot=0):
  # find the image segmentaiton using k-mean clustering algorithm 
  # input: 
  # image - the image that come from the bbox of the detector (shuld contain the balloon and the background)
  # num_cluster - the number of clusters to find in the data
  # plot -  if 1 then show the segmentation image
  # output: 
  # segmented_image - the image after the segmentation (one color for each cluster)
  # pixel values - the pixels of the after reshape (1 column vector)
  # labales - the label that given to each cluster (0,1,2..) 
  # centers - the center color of each cluster

  #####
  erorr_in_seg = 0
  # reshape the image to a 2D array of pixels and 3 color values (RGB)
  pixel_values = image.reshape((-1, 3))
  # convert to float
  pixel_values = np.float32(pixel_values)
  # define stopping criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
  # number of clusters (K)
  k = num_clusters
  try:
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
  except:
    erorr_in_seg = 1
    centers = 0
    labels = 0
    segmented_image = image
  if plot:
      plt.figure()
      plt.imshow(segmented_image)
      plt.show()
  return erorr_in_seg, segmented_image, pixel_values, labels, centers


def find_ballon_segmentation_and_params(image,num_clusters=4):
  erorr_in_seg, segmented_image, pixel_values, labels, centers = find_seg_image(image,num_clusters,0)
  if erorr_in_seg:
    radius =0 
    circle_center =(0,0);
  else:
    # find the ballon cluster
    max_cluster,center_max ,ind= find_max_cluster(pixel_values,num_clusters,labels,centers)
    # find the cog of the balloon cluster
    cog_x, cog_y, bbox_center_x, bbox_center_y = calc_cog_distance_from_bbx_center(segmented_image,center_max,0)
    # find the balloon radius
    radius, circle_center ,balloon_cog = calc_balloon_radius(segmented_image,labels,ind,0)
  return segmented_image , radius , circle_center

# segmented_image , radius , circle_center = find_ballon_segmentation_and_params(image)
# plt.imshow(segmented_image)