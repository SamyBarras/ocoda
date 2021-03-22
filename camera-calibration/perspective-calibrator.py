#! /usr/bin/env python3

"""
    This program calculates the perspective transform of the projectable area.
    The user should be able to provide appropriate camera calibration information.
    
    FROM :
    https://github.com/dclemmon/projection_mapping/blob/master/contour_perspective_calibrator.py

    MODIFIED BY SAMY BARRAS samy.barras@gmail.com
    FOR MAMATUS COLLECTIVE - 2019.11
"""

import argparse
from lxml import etree as ET

import time

import cv2
import imutils
import numpy as np
from fractions import Fraction as frac

import picamera
import pygame, sys, shutil

camResolution = (2592, 1944)
screenResolution = (1920,1080)
cameraParameters = "cameraParameters.xml"
datasPath = "cameraDatas.xml" # output file with openCV camera matrix + distortion parameters + camera settings

def displayBGColor() :
    canvas.fill((255,255,255))
    pygame.display.flip() #doesnt flip, it updates framebuffer

def load_camera_props(props_file=None):
    """
    Load the camera properties from file.  To build this file you need
    to run the aruco_calibration.py file
    :param props_file: Camera property file name
    """
    global tree, root

    if props_file is None:
        print ("load default camera param : cameraParameters.xml")
        props_file = "cameraParameters.xml"
    # load file and parse
    parser = ET.XMLParser(encoding="utf-8",remove_blank_text=True)
    fs=cv2.FileStorage(props_file,cv2.FILE_STORAGE_READ) # open cv parsing
    tree = ET.parse(props_file, parser)
    root = tree.getroot() # for standard XML parsing
    # 
    camera_matrix = fs.getNode("cameraMatrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    camResolution = np.fromstring(root.find("cameraResolution").text, dtype=int, sep=' ')
    # close file
    fs.release()

    return camera_matrix, dist_coeffs, camResolution


def undistort_image(image, camera_matrix=None, dist_coeffs=None):
    """
    Given an image from the camera module, load the camera properties and correct
    for camera distortion
    :param image: Original, distorted image
    :param camera_matrix: Param from camera calibration
    :param dist_coeffs: Param from camera calibration
    :param prop_file: The camera calibration file
    :return: Corrected image
    """
    resolution = image.shape
    if len(resolution) == 3:
        resolution = resolution[:2]
    if camera_matrix is None and dist_coeffs is None:
        camera_matrix, dist_coeffs = load_camera_props(prop_file)
    resolution = resolution[::-1]  # Shape gives us (height, width) so reverse it
    new_camera_matrix, valid_pix_roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        resolution,
        0
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        resolution,
        5
    )
    image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    return image


def find_edges(frame):
    """
    Given a frame, find the edges
    :param frame: Camera Image
    :return: Found edges in image
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Add some blur
    edged = cv2.Canny(gray, 30, 200)  # Find our edges
    return edged

def get_region_corners(frame, flipped):
    """
    Find the four corners of our projected region and return them in
    the proper order
    :param frame: Camera Image
    :return: Projection region rectangle
    """

    rect = None
    edged = find_edges(frame)

    # findContours is destructive, so send in a copy
    image, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort our contours by area, and keep the 10 largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    while True:
        screen_contours = ()
        guessedContour = 0
        for guessedContour in range(len(contours)):
            # Approximate the contour
            c = contours[guessedContour]
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 1)

            # If our contour has four points, we probably found the screen
            if len(approx) == 4:
                screen_contours = approx
                print("Four points found for screen contour")
                break
            else :
                continue
            guessedContour += 1

        if len(screen_contours) == 4 :
            cv2.drawContours(frame, [screen_contours], -1, (255, 0, 0), 1)
            wraped = convertToSurfaceImg(frame)
            displaySurfaceImage(wraped)

            try:
                user_check = str(input("Is this contour the screen ? [y/n]"))
            except ValueError:
                print("Sorry, I didn't understand that.")
                continue

            if user_check == "n":
                print("Sorry, let's try new analyse...")
                del contours[guessedContour]
                continue
            elif  user_check != "y" :
                print("Sorry, answer '", user_check,"' is not valid...")
                continue
            else :
                pts = screen_contours.reshape(4, 2)
                rect = order_corners(pts, flipped)
                print("[INFO] Contour validated by user")
                break
        else :
            wraped = convertToSurfaceImg(frame)
            displaySurfaceImage(wraped)

            try :
                user_check = str(input("Contour not found. Please make sure screen area is empty.\n-> Try again ? [y/n]"))

            except ValueError:
                print("Sorry, I didn't understand that.")
                continue
            if user_check == "n":
                print("Good bye !")
                rect = None
                break
            elif  user_check != "y" :
                print("Sorry, answer '",user_check,"' is not valid...")
                continue
            else :
                print("let's try again ,then...")
                rect = ()
                break

    # finally we return result of analyse
    return rect



def order_corners(pts, flipped=0):
    """
    Given the four points found for our contour, order them into
    Top Left, Top Right, Bottom Right, Bottom Left
    This order is important for perspective transforms
    :param pts: Contour points to be ordered correctly
    """
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    if flipped == 0 :
        rect[0] = pts[np.argmin(s)] #top left
        rect[2] = pts[np.argmax(s)] # bottom right
        rect[1] = pts[np.argmin(diff)] # top right
        rect[3] = pts[np.argmax(diff)] # bottom left
    else :
        rect[2] = pts[np.argmin(s)] #top left
        rect[0] = pts[np.argmax(s)] # bottom right
        rect[3] = pts[np.argmin(diff)] # top right
        rect[1] = pts[np.argmax(diff)] # bottom left
        
    return rect

def get_destination_array(rect):
    """
    Given a rectangle return the destination array
    :param rect: array of points  in [top left, top right, bottom right, bottom left] format
    """
    (tl, tr, br, bl) = rect  # Unpack the values
    # Compute the new image width
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # Compute the new image height
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # Our new image width and height will be the largest of each
    #max_width = max(int(width_a), int(width_b))
    #max_height = max(int(height_a), int(height_b))
    max_width = screenResolution[0]
    max_height = screenResolution[1]
    # Create our destination array to map to top-down view
    dst = np.array([
       [0, 0],  # Origin of the image, Top left
        [max_width - 1, 0],  # Top right point
        [max_width - 1, max_height - 1],  # Bottom right point
        [0, max_height - 1],  # Bottom left point
        ], dtype='float32')
    
    
    return dst, max_width, max_height


def get_perspective_transform(screen_resolution, prop_file, flipped):
    """
    Determine the perspective transform for the current physical layout
    return the perspective transform, max_width, and max_height for the
    projected region
    :param stream: Video stream from our camera
    :param screen_resolution: Resolution of projector or screen
    :param prop_file: camera property file
    """

    camera_matrix, dist_coeffs, camResolution =load_camera_props(prop_file)
    shutter = None
    awbgains = ()

    rect = None
    while True :
        #pygame.display.toggle_fullscreen()
        displayBGColor()
        # Delay execution a quarter of a second to make sure the image is displayed 
        # Don't use time.sleep() here, we want the IO loop to run.  Sleep doesn't do that
        cv2.waitKey(150)
        # Grab a photo of the frame
        with picamera.PiCamera() as camera:
            camera.resolution = camResolution
            camera.framerate = 12
            # Wait for the automatic gain control to settle
            time.sleep(1)
            # Now fix the values
            camera.shutter_speed = camera.exposure_speed
            shutter = camera.shutter_speed
            camera.exposure_mode = 'off'
            g = camera.awb_gains
            camera.awb_mode = 'off'
            camera.awb_gains = g
            awbgains = g
            #new.text = str(camera.awb_gains)
            camera.capture("persp-calibration-ref.jpg")

        camera.close()
        # back to resizable windowed app
        #pygame.display.toggle_fullscreen()
        #
        print("[INFO] Writing ref image")
        #cv2.imwrite("persp-calibration-ref.jpg", frame)
        frame = cv2.imread("persp-calibration-ref.jpg")
        # Undistort the camera image
        #cv2.waitKey(250)
        frame = undistort_image(frame, camera_matrix, dist_coeffs)
        # We're going to work with a smaller image, so we need to save the scale
        ratio = frame.shape[0] / 900.0
        orig = frame.copy()
        # Resize our image smaller, this will make things a lot faster
        frame = imutils.resize(frame, height=900)
        #cv2.waitKey(50)
        rect = get_region_corners(frame, flipped)
        if not rect is None :
            if len(rect) < 4 :
                # contour not valid, so we try again
                continue
            else :
                break
        else :
            # no contour found -> abort mission
            quit()
            break

    rect *= ratio  # We shrank the image, so now we have to scale our points up
    dst, max_width, max_height = get_destination_array(rect)

    # process correction on ref image and show result
    m = cv2.getPerspectiveTransform(rect, dst)
    wrap = cv2.warpPerspective(orig, m, (max_width, max_height))
    wraped = convertToSurfaceImg(wrap)
    displaySurfaceImage(wraped)

    # update XML file with warping datas
    # cam setup
    camSetup=ET.Element('cameraSetup')
    shut=ET.SubElement(camSetup,'shutter-speed')
    shut.text = str(shutter)
    gains=ET.SubElement(camSetup,'awb-gains')
    gains.text = '\n'.join(map(str, [' '.join(str(frac(f)).split("/")) for f in awbgains]))
    root.append(camSetup)
    # distortion parameters
    dstParam=ET.Element('distortionParam')
    ref=ET.SubElement(dstParam,'refPoints')
    ref.text = '\n'.join(map(str, [" ".join(item) for item in rect.astype(str)]))
    dest=ET.SubElement(dstParam,'destPoints')
    dest.text = '\n'.join(map(str, [" ".join(item) for item in dst.astype(str)]))
    root.append(dstParam)
    # save xml
    wrTmp(root, datasPath)
    print("xml updated")
    #
    try :
        user_check = str(input("Copy new datas to main directory ? [y/n] \n"))
    except ValueError:
        print("Sorry, I didn't understand that.")
    if user_check == "n":
        print("Ok ! Good bye !")
    elif  user_check != "y" :
        print("Sorry, answer '",user_check,"' is not valid...")
    else :
        print("OK, let's copy the file,then...")
        shutil.copy2(datasPath, "../cameraDatas.xml")
    #
    print("press esc to quit")
    return m, max_width, max_height

def wrTmp(treeObject, filepath):
    #xml_str = ('<?xml version="1.0" encoding="UTF-8"?>' + '\n' + xml.etree.ElementTree.tostring(treeObject.getroot(), method='xml'))
    xml_str = ('<?xml version="1.0" encoding="UTF-8"?>\n'.encode('utf-8') + ET.tostring(treeObject, method='xml', pretty_print=True, encoding='utf-8'))
    with open(filepath, 'wb') as xml_file:
         xml_file.write(xml_str)

def convertToSurfaceImg(cv2_img):
    wraped = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)
    img = pygame.image.frombuffer(wraped.tostring(), wraped.shape[1::-1],"RGB")
    return img

def displayImageFile(img):
    '''
        called to load image from disk
        first and only arg is image path
    '''
    image = cv2.imread(img)
    #surfImg = pygame.image.frombuffer(image.tostring(), image.shape[1::-1],"RGB")
    surfImg = convertToSurfaceImg(image)
    # image is pygame surface --> display it
    displaySurfaceImage(surfImg)
    return image

def displaySurfaceImage(image, blend=False):
    canvas.blit(image, (0, 0))
    pygame.display.flip()
    return image

def quit():
    print("Camera Datas are stored in : %s" %datasPath)
    pygame.quit()
    sys.exit()

def parse_args():
    """
    A command line argument parser
    :return:
    """
    ap = argparse.ArgumentParser()
    # camera property file
    ap.add_argument('-f', '--camera_props', default='cameraParameters.xml', help='Camera property file')
    # camera flipped
    ap.add_argument('-r', '--rotated', type=int, default=0, help='Is camera rotated ? 0 = no / 1 = mirrored')
    return vars(ap.parse_args())


if __name__ == '__main__':
    global canvas
    args = parse_args()
    #
    pygame.init()
    canvas = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    #canvas = pygame.display.set_mode((864, 648),pygame.RESIZABLE)
    #pygame.mouse.set_visible(False) # hide mouse ?
    infoObject = pygame.display.Info()
    SCREEN_WIDTH=infoObject.current_w
    SCREEN_HEIGHT=infoObject.current_h
    print("Screen size :", SCREEN_WIDTH, "-", SCREEN_HEIGHT)
    #
    try :
        user_check = str(input("Do you want to adjust the framing ? [y/n] \n"))
    except ValueError:
        print("Sorry, I didn't understand that.")
    if user_check == "n":
        print("Ok ! Let's try to find contours...")
    elif  user_check != "y" :
        print("Sorry, answer '",user_check,"' is not valid...")
    else :
        print("Adjust framing manually then press \"p\" to quit camera preview and continue.\n")
        with picamera.PiCamera() as camera:
            camera.sensor_mode=2
            camera.resolution = (2592, 1944)
            camera.start_preview()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_p :
                        camera.stop_preview()
                        #camera.close()

    #
    get_perspective_transform((SCREEN_WIDTH,SCREEN_HEIGHT), args.get('camera_props'), args.get('rotated'))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            # https://www.pygame.org/docs/ref/key.html
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_q or key == pygame.K_ESCAPE or event.scancode == 180 :
                    quit()
