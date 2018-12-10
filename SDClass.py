import cv2
import numpy as np
from PIL import ImageGrab
import time
import os
from PIL import Image, ImageDraw, ImageFont
import scipy.misc
import pickle
import datetime
# import pygame

class openCVLane():

    def __init__(self):
        self.name = "ray"

    def sobel_xy(self, img, orient='x', thresh=(20, 100)):
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
        # Return the result
        return binary_output

    # def abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
    #     gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     if orient=='x':
    #         sobel=np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    #     elif orient=='y':
    #         sobel=np.absoulte(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    #     abssobel=np.uint8(255*sobel/np.max(sobel))
    #     mask=np.zeros_like(abssobel)
    #     mask[(abssobel>=thresh_min)&(abssobel<=thresh_max)]=1
    #     return mask

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255
        # Return the binary image
        return binary_output

    # def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     # Take both Sobel x and y gradients
    #     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    #     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    #     # Calculate the gradient magnitude
    #     gradmag = np.sqrt(sobelx**2 + sobely**2)
    #     # Rescale to 8 bit
    #     scale_factor = np.max(gradmag)/255 
    #     gradmag = (gradmag/scale_factor).astype(np.uint8) 
    #     # Create a binary image of ones where threshold is met, zeros otherwise
    #     binary_output = np.zeros_like(gradmag)
    #     binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    #     # Return the binary image
    #     return binary_output


    def dir_thresh(self, img, sobel_kernel=3, thresh=(0.7, 1.3)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255
        # Return the binary image
        return binary_output.astype(np.uint8)

    # def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
    #     gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #     sobelx=np.absolute(cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=sobel_kernel))
    #     sobely=np.absolute(cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=sobel_kernel))
    #     theta=np.arctan2(sobely, sobelx)
    #     binary_output=np.zeros_like(theta)
    #     binary_output[(theta>=thresh[0]) & (theta<=thresh[1])]=1
    #     # Remove this line
    #     return binary_output



    def gradient_combine(self, img, th_x, th_y, th_mag, th_dir):
        # Find lane lines with gradient information of Red channel
        rows, cols = img.shape[:2]
        R = img[220:rows - 12, 0:cols, 2]

        sobelx = self.sobel_xy(R, 'x', th_x)
        #cv2.imshow('sobel_x', sobelx)
        sobely = self.sobel_xy(R, 'y', th_y)
        #cv2.imshow('sobel_y', sobely)
        mag_img = self.mag_thresh(R, 3, th_mag)
        #cv2.imshow('sobel_mag', mag_img)
        dir_img = self.dir_thresh(R, 15, th_dir)
        #cv2.imshow('result5', dir_img)

        # combine gradient measurements
        gradient_comb = np.zeros_like(dir_img).astype(np.uint8)
        gradient_comb[((sobelx > 1) & (mag_img > 1) & (dir_img > 1)) | ((sobelx > 1) & (sobely > 1))] = 255
        return gradient_comb


    # def hls_select(self, img, thresh=(0, 255)):
    #     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #     s_channel = hls[:,:,2]
    #     binary_output = np.zeros_like(s_channel)
    #     binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    #     return binary_output

    def ch_thresh(self, ch, thresh=(80, 255)):
        binary = np.zeros_like(ch)
        binary[(ch > thresh[0]) & (ch <= thresh[1])] = 255
        return binary

    def hls_combine(self, img, th_h, th_l, th_s):
        # convert to hls color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        rows, cols = img.shape[:2]
        R = img[220:rows - 12, 0:cols, 2]
        _, R = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
        #cv2.imshow('red!!!',R)
        H = hls[220:rows - 12, 0:cols, 0]
        L = hls[220:rows - 12, 0:cols, 1]
        S = hls[220:rows - 12, 0:cols, 2]

        h_img = self.ch_thresh(H, th_h)
        #cv2.imshow('HLS (H) threshold', h_img)
        l_img = self.ch_thresh(L, th_l)
        #cv2.imshow('HLS (L) threshold', l_img)
        s_img = self.ch_thresh(S, th_s)
        #cv2.imshow('HLS (S) threshold', s_img)

        # Two cases - lane lines in shadow or not
        hls_comb = np.zeros_like(s_img).astype(np.uint8)
        hls_comb[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255 # | (R > 1)] = 255
        #hls_comb[((s_img > 1) & (h_img > 1)) | (R > 1)] = 255
        return hls_comb

    def comb_result(self, grad, hls):
        """ give different value to distinguish them """
        result = np.zeros_like(hls).astype(np.uint8)
        #result[((grad > 1) | (hls > 1))] = 255
        result[(grad > 1)] = 100
        result[(hls > 1)] = 255

        return result


    # Edit this function to create your own pipeline.
    def pipeline(self, img, s_thresh=(175, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        # Convert to hls color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
        return color_binary

    def warp_image(self, img, src, dst, size):
        # Perspective Transform
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
        return warp_img, M, Minv


    # def unwarp(self, img, src, dst):
    #     h,w = img.shape[:2]
    #     # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    #     M = cv2.getPerspectiveTransform(src, dst)
    #     Minv = cv2.getPerspectiveTransform(dst, src)
    #     # use cv2.warpPerspective() to warp your image to a top-down view
    #     warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    #     return warped, M, Minv


    def rad_of_curvature(self, left_line, right_line):
        # measure radius of curvature

        ploty = left_line.ally
        leftx, rightx = left_line.allx, right_line.allx

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Define conversions in x and y from pixels space to meters
        width_lanes = abs(right_line.startx - left_line.startx)
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7*(720/1280) / width_lanes  # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # radius of curvature result
        left_line.radius_of_curvature = left_curverad
        right_line.radius_of_curvature = right_curverad

    def smoothing(self, lines, pre_lines=3):
        # collect lines & print average line
        lines = np.squeeze(lines)
        avg_line = np.zeros((720))

        for ii, line in enumerate(reversed(lines)):
            if ii == pre_lines:
                break
            avg_line += line
        avg_line = avg_line / pre_lines

        return avg_line

    def blind_search(self, b_img, left_line, right_line):

        # blind search - first frame, lost lane lines
        # using histogram & sliding window

        # Take a histogram of the bottom half of the image
        histogram = np.sum(b_img[int(b_img.shape[0] / 2):, :], axis=0)

        # Create an output image to draw on and  visualize the result
        output = np.dstack((b_img, b_img, b_img)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        start_leftX = np.argmax(histogram[:midpoint])
        start_rightX = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        num_windows = 9
        # Set height of windows
        window_height = np.int(b_img.shape[0] / num_windows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = b_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        current_leftX = start_leftX
        current_rightX = start_rightX

        # Set minimum number of pixels found to recenter window
        min_num_pixel = 50

        # Create empty lists to receive left and right lane pixel indices
        win_left_lane = []
        win_right_lane = []

        window_margin = left_line.window_margin

        # Step through the windows one by one
        for window in range(num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = b_img.shape[0] - (window + 1) * window_height
            win_y_high = b_img.shape[0] - window * window_height
            win_leftx_min = current_leftX - window_margin
            win_leftx_max = current_leftX + window_margin
            win_rightx_min = current_rightX - window_margin
            win_rightx_max = current_rightX + window_margin

            # Draw the windows on the visualization image
            cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
                nonzerox <= win_leftx_max)).nonzero()[0]
            right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
                nonzerox <= win_rightx_max)).nonzero()[0]
            # Append these indices to the lists
            win_left_lane.append(left_window_inds)
            win_right_lane.append(right_window_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(left_window_inds) > min_num_pixel:
                current_leftX = np.int(np.mean(nonzerox[left_window_inds]))
            if len(right_window_inds) > min_num_pixel:
                current_rightX = np.int(np.mean(nonzerox[right_window_inds]))

        # Concatenate the arrays of indices
        win_left_lane = np.concatenate(win_left_lane)
        win_right_lane = np.concatenate(win_right_lane)

        # Extract left and right line pixel positions
        leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
        rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

        output[lefty, leftx] = [255, 0, 0]
        output[righty, rightx] = [0, 0, 255]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_line.current_fit = left_fit
        right_line.current_fit = right_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

        # ax^2 + bx + c
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        left_line.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)

        if len(left_line.prevx) > 10:
            left_avg_line = self.smoothing(left_line.prevx, 10)
            left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
            left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
            left_line.current_fit = left_avg_fit
            left_line.allx, left_line.ally = left_fit_plotx, ploty
        else:
            left_line.current_fit = left_fit
            left_line.allx, left_line.ally = left_plotx, ploty

        if len(right_line.prevx) > 10:
            right_avg_line = self.smoothing(right_line.prevx, 10)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
            right_line.current_fit = right_avg_fit
            right_line.allx, right_line.ally = right_fit_plotx, ploty
        else:
            right_line.current_fit = right_fit
            right_line.allx, right_line.ally = right_plotx, ploty

        left_line.startx, right_line.startx = left_line.allx[len(left_line.allx)-1], right_line.allx[len(right_line.allx)-1]
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

        left_line.detected, right_line.detected = True, True
        # print radius of curvature
        self.rad_of_curvature(left_line, right_line)
        return output

    def prev_window_refer(self, b_img, left_line, right_line):

        # refer to previous window info - after detecting lane lines in previous frame
        # Create an output image to draw on and  visualize the result
        output = np.dstack((b_img, b_img, b_img)) * 255

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = b_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set margin of windows
        window_margin = left_line.window_margin

        left_line_fit = left_line.current_fit
        right_line_fit = right_line.current_fit
        leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
        leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
        rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
        rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

        # Identify the nonzero pixels in x and y within the window
        left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
        right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

        # Extract left and right line pixel positions
        leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
        rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

        output[lefty, leftx] = [255, 0, 0]
        output[righty, rightx] = [0, 0, 255]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

        # ax^2 + bx + c
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        leftx_avg = np.average(left_plotx)
        rightx_avg = np.average(right_plotx)

        left_line.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)

        if len(left_line.prevx) > 10:
            left_avg_line = self.smoothing(left_line.prevx, 10)
            left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
            left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
            left_line.current_fit = left_avg_fit
            left_line.allx, left_line.ally = left_fit_plotx, ploty
        else:
            left_line.current_fit = left_fit
            left_line.allx, left_line.ally = left_plotx, ploty

        if len(right_line.prevx) > 10:
            right_avg_line = self.smoothing(right_line.prevx, 10)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
            right_line.current_fit = right_avg_fit
            right_line.allx, right_line.ally = right_fit_plotx, ploty
        else:
            right_line.current_fit = right_fit
            right_line.allx, right_line.ally = right_plotx, ploty

        # goto blind_search if the standard value of lane lines is high.
        standard = np.std(right_line.allx - left_line.allx)

        if (standard > 80):
            left_line.detected = False

        left_line.startx, right_line.startx = left_line.allx[len(left_line.allx) - 1], right_line.allx[len(right_line.allx) - 1]
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

        # print radius of curvature
        self.rad_of_curvature(left_line, right_line)
        return output

    def find_LR_lines(self, binary_img, left_line, right_line):
        # find left, right lines & isolate left, right lines
        # blind search - first frame, lost lane lines
        # previous window - after detecting lane lines in previous frame

        # if don't have lane lines info
        if left_line.detected == False:
            return self.blind_search(binary_img, left_line, right_line)
        # if have lane lines info
        else:
            return self.prev_window_refer(binary_img, left_line, right_line)

    def draw_lane(self, img, left_line, right_line, lane_color=(255, 0, 255), road_color=(0, 255, 0)):
        # draw lane lines & current driving space
        window_img = np.zeros_like(img)

        window_margin = left_line.window_margin
        left_plotx, right_plotx = left_line.allx, right_line.allx
        ploty = left_line.ally

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_pts_l = np.array([np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
        left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
        left_pts = np.hstack((left_pts_l, left_pts_r))
        right_pts_l = np.array([np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
        right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
        right_pts = np.hstack((right_pts_l, right_pts_r))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
        cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([pts]), road_color)
        result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

        return result, window_img

    def road_info(self, left_line, right_line):
        # print road information onto result image
        curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2

        direction = ((left_line.endx - left_line.startx) + (right_line.endx - right_line.startx)) / 2

        if curvature > 2000 and abs(direction) < 100:
            road_inf = 'No Curve'
            curvature = -1
        elif curvature <= 2000 and direction < - 50:
            road_inf = 'Left Curve'
        elif curvature <= 2000 and direction > 50:
            road_inf = 'Right Curve'
        else:
            if left_line.road_inf != None:
                road_inf = left_line.road_inf
                curvature = left_line.curvature
            else:
                road_inf = 'None'
                curvature = curvature

        center_lane = (right_line.startx + left_line.startx) / 2
        lane_width = right_line.startx - left_line.startx

        center_car = 720 / 2
        if center_lane > center_car:
            deviation = 'Left ' + str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
        elif center_lane < center_car:
            deviation = 'Right ' + str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
        else:
            deviation = 'Center'
        left_line.road_inf = road_inf
        left_line.curvature = curvature
        left_line.deviation = deviation

        return road_inf, curvature, deviation

    def print_road_status(self, img, left_line, right_line):
        # print road status (curve direction, radius of curvature, deviation)
        road_inf, curvature, deviation = self.road_info(left_line, right_line)
        cv2.putText(img, 'Road Status', (22, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (80, 80, 80), 2)

        lane_inf = 'Lane Info : ' + road_inf
        if curvature == -1:
            lane_curve = 'Curvature : Straight line'
        else:
            lane_curve = 'Curvature : {0:0.3f}m'.format(curvature)
        deviate = 'Deviation : ' + deviation

        cv2.putText(img, lane_inf, (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.putText(img, lane_curve, (10, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.putText(img, deviate, (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        return img

    def print_road_map(self, image, left_line, right_line):
        # print simple road map
        img = cv2.imread('top_view_car.png', -1)
        img = cv2.resize(img, (120, 246))

        rows, cols = image.shape[:2]
        window_img = np.zeros_like(image)

        window_margin = left_line.window_margin
        left_plotx, right_plotx = left_line.allx, right_line.allx
        ploty = left_line.ally
        lane_width = right_line.startx - left_line.startx
        lane_center = (right_line.startx + left_line.startx) / 2
        lane_offset = cols / 2 - (2*left_line.startx + lane_width) / 2
        car_offset = int(lane_center - 360)
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_pts_l = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width - window_margin / 4, ploty]))])
        left_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - lane_width+ window_margin / 4, ploty])))])
        left_pts = np.hstack((left_pts_l, left_pts_r))
        right_pts_l = np.array([np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty]))])
        right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset + window_margin / 4, ploty])))])
        right_pts = np.hstack((right_pts_l, right_pts_r))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_pts]), (140, 0, 170))
        cv2.fillPoly(window_img, np.int_([right_pts]), (140, 0, 170))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width + window_margin / 4, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

        #window_img[10:133,300:360] = img
        road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
        window_img = Image.fromarray(window_img)
        img = Image.fromarray(img)
        road_map.paste(window_img, (0, 0))
        road_map.paste(img, (300-car_offset, 590), mask=img)
        road_map = np.array(road_map)
        road_map = cv2.resize(road_map, (95, 95))
        road_map = cv2.cvtColor(road_map, cv2.COLOR_BGRA2BGR)
        return road_map

    def fitlines(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        # Fit a second order polynomial to each
        if len(leftx) == 0:
            left_fit =[]
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
        
        if len(rightx) == 0:
            right_fit =[]
        else:
            right_fit = np.polyfit(righty, rightx, 2)
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        return left_fit, right_fit,out_img


    def fit_continuous(self, left_fit, right_fit, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        if len(leftx) == 0:
            left_fit_updated =[]
        else:
            left_fit_updated = np.polyfit(lefty, leftx, 2)
        
        if len(rightx) == 0:
            right_fit_updated =[]
        else:
            right_fit_updated = np.polyfit(righty, rightx, 2)
            
        return  left_fit_updated, right_fit_updated



    #Calc Curvature
    def curvature(self, left_fit, right_fit, binary_warped):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        y_eval = np.max(ploty)
        
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension


        # Fit new polynomials to x,y in world space
        #leftx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        #rightx = right_fit[0]*ploty**2+right_fit[1]*ploty+left_fit[2]
            
        #left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        #right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        center = (((left_fit[0]*720**2+left_fit[1]*720+left_fit[2]) +(right_fit[0]*720**2+right_fit[1]*720+right_fit[2]) ) /2 - 640)*xm_per_pix
        
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm', right_curverad, 'm')
        return left_curverad, right_curverad, center




    def roi(self, img, vertices):
        # #blank mask:
        # mask = np.zeros_like(img)
        # # fill the mask
        # cv2.fillPoly(mask, vertices, 255)

        # # now only show the area that is the mask
        # masked = cv2.bitwise_and(img, mask)
        # return masked

        height,width,depth = img.shape
        maskImage = np.zeros((height,width), np.uint8)
        cv2.fillPoly(maskImage, vertices, (255,255,255))
        # cv2.rectangle(maskImage, (30, 30), (500, 400), (255,255,255), thickness=-1)
        masked_data = cv2.bitwise_and(img, img, mask=maskImage)
        return masked_data


    # convert MP4 size
    def convertMP4Size(self, srcFileName, detFileName, detSize):
        videoCapture = cv2.VideoCapture(srcFileName)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        videoWriter = cv2.VideoWriter(detFileName, fourcc, fps, detSize)

        if(videoCapture.isOpened==False):
            print("Error opening Video strearm")
        else:
            while(videoCapture.isOpened()):
                ret, frame=videoCapture.read()
                if(ret==True):
                    frame=cv2.resize(frame,detSize)
                    cv2.waitKey(1) #延迟
                    cv2.imshow("Oto Video", frame) #显示
                    videoWriter.write(frame) #写视频帧
                else:
                    break
        
        cv2.destroyAllWindows()
        videoCapture.read()



    # slice video
    def sliceVideo(self, srcFileName, detFileName, sliceFromat):
        videoCapture = cv2.VideoCapture(srcFileName)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        videoWriter = cv2.VideoWriter(detFileName, fourcc, fps, ((sliceFromat[3]-sliceFromat[2]),(sliceFromat[1]-sliceFromat[0])))

        if(videoCapture.isOpened==False):
            print("Error opening Video strearm")
        else:
            while(videoCapture.isOpened()):
                ret, frame=videoCapture.read()
                if(ret==True):
                    sliceImg = frame[sliceFromat[0]:sliceFromat[1], sliceFromat[2]:sliceFromat[3]]
                    cv2.waitKey(1) #延迟
                    cv2.imshow("Oto Video", sliceImg) #显示
                    videoWriter.write(sliceImg) #写视频帧
                else:
                    break
        
        cv2.destroyAllWindows()
        videoCapture.read()
    
    # capture video clips at certain area of screen
    def screen_record(self,scrLocSize): 
        last_time = time.time()
        while(True):
            # 800x600 windowed mode
            printscreen =  np.array(ImageGrab.grab(bbox=scrLocSize))
            # printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    
    def drawCurveLine(self, img, degrees):
        height, width=img.shape
        curve=int(abs(degrees))
        if(degrees>0):
            cv2.ellipse(img, (int(width/2)+int(degrees), height), (curve, int(height/2)), 0, 180, 270, (255,0,0), 2)
        else:
            cv2.ellipse(img, (int(width/2)+int(degrees), height), (curve, int(height/2)), 0, 270, 360, (255,0,0), 2)
        return img

    def showNPZDataByName(self, fileName):
        data = np.load(fileName)
        print(data['train'].shape)
        print(data['train_labels'])

        cv2.imshow("npz", cv2.cvtColor(data['train'], cv2.COLOR_RGB2BGR))
        # cv2.imshow("image", data['train'])
        cv2.waitKey()
        data.close()


    def saveJPGWithLable(self, imageData, labelData, dir, fileName):
        if not os.path.exists(dir):
            os.makedirs(dir)
        try:
            width=imageData.shape[1]
            height=imageData.shape[0]
            # print('w:{},h:{}'.format(width,height))

            curve=int(abs(labelData))
            if(labelData>0):
                cv2.ellipse(imageData, (int(width/2)+int(labelData), height), (curve, int(height/2)), 0, 180, 270, (255,0,0), 2)
            else:
                cv2.ellipse(imageData, (int(width/2)+int(labelData), height), (curve, int(height/2)), 0, 270, 360, (255,0,0), 2)

            im = Image.fromarray(imageData)
            draw = ImageDraw.Draw(im)
            myfont = ImageFont.truetype('C:/windows/fonts/Arial.ttf', size=20)
            fillcolor = "#ffffff"

            draw.text((width-50, 5), str(labelData), font=myfont, fill=fillcolor)
            im.save(dir + '/' + fileName + '.jpg')

        except IOError as e:
            print(e)

    def saveJPGForTrain(self, imageData, dir, fileName):
        if not os.path.exists(dir):
            os.makedirs(dir)
        try:
            # width=imageData.shape[1]
            # height=imageData.shape[0]
            # print('w:{},h:{}'.format(width,height))

            # raw image
            im = Image.fromarray(imageData)

            # resize image
            # im = Image.fromarray(scipy.misc.imresize(imageData, [256, 455]))
            im.save(dir + '/' + fileName + '.jpg')

        except IOError as e:
            print(e)

    def convertNPZ2JpgByDir(self, dirName):
        filenames = os.listdir(dirName)
        for filename in filenames:
            if(filename.endswith('.npz')):
                print(filename)
                data = np.load(dirName+'/'+filename)
                print(data['train_labels'][0])
                showLable=data['train_labels'][0]
                self.saveJPGWithLable(data['train'], showLable, dirName+'/'+'jpgWithLable', filename)
                data.close()
    

    def convertNPZ2Jpg2TrianByDir(self, dirName):
        if not os.path.exists(dirName+'/jpgForTrain'):
            os.makedirs(dirName+'/jpgForTrain')
        with open(dirName+'/jpgForTrain/data.txt', "a") as f:
            filenames = os.listdir(dirName)
            for filename in filenames:
                if(filename.endswith('.npz')):
                    print(filename)
                    data = np.load(dirName+'/'+filename)
                    showLable=data['train_labels'][0]
                    showLable=showLable*4
                    print(showLable)
                    f.write(filename+".jpg "+str(showLable)+"\n")
                    self.saveJPGForTrain(data['train'], dirName+'/'+'jpgForTrain', filename)
                    data.close()

    def calib(self, calibConfig):
        dist_pickle = pickle.load(open(calibConfig, "rb" ))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        return mtx, dist



class FindLine():

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Set the width of the windows +/- margin
        self.window_margin = 56
        # x values of the fitted line over the last n iterations
        self.prevx = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_inf = None
        self.curvature = None
        self.deviation = None



class YOLOV3():

    def __init__(self):

        self.classes = None
        self.cfg = None
        self.weights = None
        self.COLORS = None
        self.scale = 0.00392
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.scrWidth=0
        self.scrHeight=0
        self.showObjIDs=[0,1,2,3,5,6,7,9,10,11,12,13]


    def load_cfg_weight_classTXT(self, cfg, weights, classTXT):
        with open(classTXT, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.cfg = cfg
        self.weights = weights
        self.net = cv2.dnn.readNet(self.weights, self.cfg)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def detectObjs(self, image):

        oldtime=datetime.datetime.now()

        if(self.scrWidth==0 or self.scrHeight==0):
            self.scrWidth = image.shape[1]
            self.scrHeight = image.shape[0]

        blob = cv2.dnn.blobFromImage(image, self.scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers(self.net))

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)

                # showObjIDs
                # if(class_id in self.showObjIDs):

                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * self.scrWidth)
                    center_y = int(detection[1] * self.scrHeight)
                    w = int(detection[2] * self.scrWidth)
                    h = int(detection[3] * self.scrHeight)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        cv2.imshow("object detection", image)
        newtime=datetime.datetime.now()
        print('yolo用时:%s秒'%(newtime-oldtime))



    def returnObjs(self, image):
        oldtime=datetime.datetime.now()
        if(self.scrWidth==0 or self.scrHeight==0):
            self.scrWidth = image.shape[1]
            self.scrHeight = image.shape[0]
        blob = cv2.dnn.blobFromImage(image, self.scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers(self.net))
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                # showObjIDs
                if(class_id in self.showObjIDs):
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * self.scrWidth)
                        center_y = int(detection[1] * self.scrHeight)
                        w = int(detection[2] * self.scrWidth)
                        h = int(detection[3] * self.scrHeight)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        newtime=datetime.datetime.now()
        print('yolo用时:%s秒'%(newtime-oldtime))
        return class_ids, indices, confidences, boxes


    def drawObjs(self, img, class_ids, indices, confidences, boxes):
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        return img

