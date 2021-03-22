#!/full/path/to/example-env/bin/python2
"""
    This program is developped by Samy Barras
    For MAMATUS COLLECTIVE - 2019.11


"""
import sys, os, time, shutil
import RPi.GPIO as GPIO
import picamera
from picamera.array import *
from datetime import datetime
import pygame
from pygame.locals import * 
import cv2
import numpy as np
from lxml import etree as ET
from threading import Thread, Timer
import skimage
from skimage import data, exposure, img_as_float
from fractions import Fraction as frac
import argparse
import collections
# to run through ssh: export DISPLAY=:x.y (define x,y with echo $DISPLAY)

# CUSTOM VARIABLES  #############################################
SPLASHSCREEN_TIME = 3000  # time in milliseconds for showing splash-screen
USE_MOTIONDETECTION = False # hazardous (not working) --> detect motion to get in/out of instructions mode
NOACTIVITY_TIME = 300.0 # time in seconds without activity before showing instructions
BG_COLOR = (255, 255, 255)
CORNERPIN_STEP = 3
FPS = 12.0
ONIONSKIN = 3 # int / num of frames to show as onion skin
# colors
WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
BLACK =     (  0,   0,   0)
# GPIO Inputs (BCM) & OUTPUTS
SHOT_BUTTON = 18
PLAY_BUTTON = 21
SHOT_LED = 25
# CONSTANT VARIABLES #############################################
frame_counter=0
fs = None # cameraParameters data (calibration / xml)
canvas = None #drawing surface
is_playing=False
cornerPinMode = False
SCREEN_WIDTH=0
SCREEN_HEIGHT=0
VALID_EXT = ['png','PNG']
'''
    LAST_DISPLAY tracks what was last image in background :
    0 -> empty
    1 -> instructions
    2 -> last frame
'''
LAST_DISPLAY = 0
ACTIVITY_STATUS = 0  # 0 --> no activity since a while   1 --> activity detected
IS_SHOOTING = False
USE_GABARIT = False

# ANIMATION #############################################
def captureFrame(): #channel
    global frame_counter, IS_SHOOTING

    updateTimer () # for activity check
    IS_SHOOTING = True
    # display capture background (empty frame)
    canvas.fill(BG_COLOR)
    pygame.display.flip() 
    cv2.waitKey(30)
    # setup files name & path
    takeName = str(frame_counter).zfill(5) + ".png"
    filepath = working_dir + '/' + takeName
    fullResFilePath = working_dir + '/full-res-img/' + takeName
    # empty image buffer
    image = None
    # take picture
    #start = time.time()
    full, wrap = mycamera.customCaptureAndProcess()
    #end = time.time()
    #print("take took :", end-start,"seconds")
    IS_SHOOTING = False
    displaySurfaceImage()
    # write files
    cv2.imwrite(fullResFilePath, cv2.cvtColor(full,cv2.COLOR_BGR2RGB))
    cv2.imwrite(filepath, cv2.cvtColor(wrap,cv2.COLOR_BGR2RGB))
    # print message to confirm frame is captured and saved   
    print("Frame ", frame_counter, " captured.")
    # update counter once everything is done
    frame_counter+=1


def convertToSurfaceImg(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = pygame.image.frombuffer(img.tostring(), img.shape[1::-1],"RGB")
    return img

def displayImageFile(img):
    '''
        called to load image from disk
        first and only arg is image path
    '''
    image = cv2.imread(img)
    # image is in cv2 BGR mode -> convert to pygame surface --> display it
    displaySurfaceImage(convertToSurfaceImg(image))
    return image

def displaySurfaceImage(image=None, blend=True):
    # show onion skin
    frame = None
    if image is None :
        frame = frames[-1]
    else :
        frame = image.copy()

    if ONIONSKIN >= 1 :
        # we always show last frame with full aplha
        image = pygame.transform.scale(frame, (SCREEN_WIDTH,SCREEN_HEIGHT))
        canvas.blit(image, (0, 0))

    if ONIONSKIN > 1 :
        # if onion skin set, we show previous frames with less opacity
        f = ONIONSKIN
        while f > 0 :
            if len(frames) >= f :
                alpha = 255/int(f+1)
                print("-",f," --> ", alpha)
                frame = frames[-f]
                image = pygame.transform.scale(frame, (SCREEN_WIDTH,SCREEN_HEIGHT))
                #image = image.convert_alpha()
                image.set_alpha(alpha)
                canvas.blit(image, (0, 0))
            else :
                pass
            f -= 1

    # show gabarit
    if USE_GABARIT and blend == True :
        canvas.blit(gabarit, (0, 0), special_flags=pygame.BLEND_MULT)

    #update screen
    pygame.display.flip()
    updateLastDisplay(2)
    return image

def displayLastFrame():
    '''
        this function display last frame of animation in background
    '''
    #Get the latest image file in the working directory
    valid_files = [os.path.join(working_dir, filename) for filename in os.listdir(working_dir)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and \
        f.rsplit('.',1)[-1] in VALID_EXT and os.path.isfile(f)]

    if not valid_files:
        # working dir is empty --> show empty frame
        displayBGColor()
    else :
        last_file = max(valid_files, key=os.path.getmtime)
        #print(last_file)
        #[int(s) for s in last_file.split() if s.isdigit()]
        displayImageFile(last_file)

def displayLastTake():
    if len(frames) > 0 :
        displaySurfaceImage(frames[-1])

def displayBGColor() :
    canvas.fill(BG_COLOR)
    if USE_GABARIT :
        canvas.blit(gabarit, (0, 0))
    pygame.display.flip() #doesnt flip, it updates framebuffer
    updateLastDisplay(0)

def displayEmpty():
    # called only when taking a picture
    canvas.fill(BG_COLOR)
    pygame.display.flip() #doesnt flip, it updates framebuffer

def displayText(text):
    canvas.fill(BG_COLOR)
    font_big = pygame.font.Font(None, 80)
    color_white=(0,0,0)
    text_surface = font_big.render('%s'%text, True, color_white)
    x, y = SCREEN_WIDTH/2,SCREEN_HEIGHT/2 #centering
    canvas.blit(text_surface, (x - text_surface.get_width() // 2, y - text_surface.get_height() // 2 ))
    pygame.display.flip() #doesnt flip, it updates framebuffer

    # update last_display value to match instructions screens
    updateLastDisplay(1)

def displaySpashScreen(imagepath):
    im = cv2.imread(imagepath)
    wrap = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    img = pygame.image.frombuffer(wrap.tostring(), wrap.shape[1::-1],"RGB") #wrap.shape[1::-1]
    image = pygame.transform.scale(img, (SCREEN_WIDTH, SCREEN_HEIGHT))
    canvas.blit(image, (0, 0))
    pygame.display.flip()
    # displaySurfaceImage(img, blend=False)
    # update last_display value to match instructions screens
    updateLastDisplay(1)

def playBufferAnimation(channel=0): #channel
    global is_playing, clock
    updateTimer () # for activity check
    print("[INFO] Play animation")

    if not(is_playing): #block animation event while animation is running
        is_playing=True
        while True :
            for img in frames:
                start = time.time()
                #these 2 operations take between 0.07 and 0.11 secs, mean 0.1 sec. No FPS above 10
                canvas.blit(img, (0, 0)) #Replace (0, 0) with desired coordinates
                if USE_GABARIT :
                    canvas.blit(gabarit, (0, 0), special_flags=pygame.BLEND_MULT)
                pygame.display.flip()
                end = time.time()
                print("display take:", end-start,"seconds")
            is_playing=False
            break
    displaySurfaceImage()
    #displayLastTake()


def compileAnimation(): #channel
    print("Compiling...")
    os.system("avconv -r 12 -i "+ working_dir+"/%03d.jpg"+" -qscale 2 animation.h264")
    print("Animation compiled.")

def displayVideoStream ():
    '''
    optional function to make the focus
    '''
    #camera.annotate_background = picamera.Color('black')
    mycamera.annotate_text = "test of string"
    mycamera.start_preview(resolution=(1920,1444))

    while True:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                key = event.key
                if key == pygame.K_f:
                    mycamera.stop_preview()
                    break

def updateLastDisplay(d):
    global LAST_DISPLAY
    LAST_DISPLAY = d
    ##print("[INFO] last image displayed in background is: %s" %d)

# USER ACTIVITY #############################################
def blinkLed ():
    global IS_SHOOTING
    while True :
        if IS_SHOOTING == True :
            GPIO.output(25,GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(25,GPIO.LOW)
            time.sleep(0.3)
        else :
            # set led output to LOW if it is HIGH
            if GPIO.input(25) == 1 :
                GPIO.output(25,GPIO.LOW)

def actionButtn(inputbttn):
    '''
    function called each time a button is pressed
    will define to shot a frame / play anim / or get out of wainting screen
    '''
    updateTimer()
    if ACTIVITY_STATUS != 0 :
        if inputbttn == SHOT_BUTTON and GPIO.input(inputbttn) == 0:
            #print ("shot")
            captureFrame()
        elif inputbttn == PLAY_BUTTON and GPIO.input(inputbttn) == 0:
            #print("play")
            playBufferAnimation()
    else :
        #print("instructions screened")
        if GPIO.input(inputbttn) == 0 :
            #print("updateTimer")
            activityCheck()

    return # not needed, just for clarity

def updateTimer ():
    '''
    simple function to update a timer at each action done by user
    '''
    global timer
    timer = time.time()
    

def updateActStatus(s) :
    global ACTIVITY_STATUS
    #print("update activity status to %s" %s)
    ACTIVITY_STATUS = s
    if ACTIVITY_STATUS == 0 :
        # no activity since X seconds
        displaySpashScreen("resources/instructions.jpg")
    else :
        # activity detected X seconds ago
        displayLastFrame()

def motionAnalyse(gray, avg) :
    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    frameDelta = cv2.absdiff(cv2.convertScaleAbs(avg), gray)
    # threshold the delta image
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    if thresh.sum() > 100 :
        # motion detected while instructions are showed
        #if ACTIVITY_STATUS == 0 : # instruction
        print("motion detected")
        updateTimer() # update timer each time motion is detected


avg = None
def detectMotion ():
    global mycamera, avg, ACTIVITY_STATUS
    print ("recording for motion analysis")
    with picamera.array.PiRGBArray(mycamera, size=(320,240)) as rawCapture:
        for f in mycamera.capture_continuous(rawCapture, format="rgb", resize=(320,240), use_video_port=True):
            # grab the raw NumPy array representing the image and initialize
            # the timestamp and occupied/unoccupied text
            frame = f.array
            #rawCapture.truncate(0)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            # if the average frame is None, initialize it
            if avg is None:
                avg = gray.copy().astype("float")
                rawCapture.truncate(0)
                continue
            
            # analyse motion between two frames
            motionAnalyse(gray, avg)

            # Once motion detection is done, make the prior image the current
            avg = gray.copy().astype("float")
            rawCapture.truncate(0)
    


def activityCheck():
    global timer, NOACTIVITY_TIME, LAST_DISPLAY, ACTIVITY_STATUS
    ''' 
    this thread is launched every X seconds
    it check time elapsed since last saved timer
    if elapsed time is > X seconds it means no activity since then,
    so we show instructions
    '''
    thrd = Timer(10, activityCheck)
    thrd.daemon = True
    thrd.start()

    t = time.time()-timer
    if (time.time()-timer) >= NOACTIVITY_TIME and ACTIVITY_STATUS != 0 :
        #print("-- no activity since a while")
        updateActStatus(0)
    elif (time.time()-timer) < NOACTIVITY_TIME and ACTIVITY_STATUS != 1 :
        updateActStatus(1)
    '''
    if (time.time()-timer) >= NOACTIVITY_TIME:
        # no activity since a while !
        print ("-- no activity since %f --" %t)
        if LAST_DISPLAY != 1 :
            print("show instructions")
            displaySpashScreen("resources/instructions.jpg")

        cv2.waitKey(250)
        # wait a bit of time before capturing motion --> to avoid capturing motion when showing instructions screen
        print("wait a bit before analysing motion  -- or press any key")
        updateActStatus(0)

    else :
        print("activity detetcted")
        if LAST_DISPLAY == 1 :
            displayLastFrame()
        elif ACTIVITY_STATUS == 0 :
            updateActStatus(1)
    '''

# POPUP MENU
def make_popup():
    popupSurf = pygame.Surface((200,200))
    options = ['Background color',
               'corner pin tool',
               'color correction'
               'help / credits']
    for i in range(len(options)):
        textSurf = FONT.render(options[i], 1, BLUE)
        textRect = textSurf.get_rect()
        textRect.top = (SCREEN_HEIGHT/2)-100
        textRect.left = (SCREEN_WIDTH/2)-100
        #self.top += pygame.font.Font.get_linesize(FONT)
        popupSurf.blit(textSurf, textRect)
    popupRect = popupSurf.get_rect()
    popupRect.centerx = SCREEN_WIDTH/2
    popupRect.centery = SCREEN_HEIGHT/2
    canvas.blit(popupSurf, popupRect)
    pygame.display.update()



# CAMERA THREADIND #########################################
class MyCamera(picamera.PiCamera):
    def __init__(self, resolution, framerate, sensor_mode, **kwargs):
        super(MyCamera, self).__init__()
        self.resolution = resolution
        self.framerate = framerate
        self.sensor_mode = sensor_mode
        # set optional camera parameters (refer to PiCamera docs)
        for (arg, value) in kwargs.items():
            setattr(self.camera, arg, value)

        # capturing consistent images recipe
        # Set ISO to the desired value
        self.iso = 400
        # Wait for the automatic gain control to settle
        time.sleep(2)
        # Now fix the values
        self.shutter_speed = self.exposure_speed
        self.exposure_mode = 'off'
        g = self.awb_gains
        self.awb_mode = 'off'
        self.awb_gains = g

        # setup custom encoder
        self.output = PiYUVArray(self, size=self.resolution)
        self.lastframe = None


    def undistort(self, image):
        # setup of cv2 distortion matrices
        resolution = image.shape
        if len(resolution) == 3 :
            resolution = resolution[:2]
        resolution = resolution[::-1]  # Shape gives us (height, width) so reverse it
        #
        new_camera_matrix, valid_pix_roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            resolution,
            0
        )
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix,
            resolution,
            5
        )
        image = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        return image

    def postProcess(self, image):
        #image warping
        m = cv2.getPerspectiveTransform(refPoints, destPoints)
        wrap = cv2.warpPerspective(image, m, (SCREEN_WIDTH, SCREEN_HEIGHT))
        imgSurface = pygame.image.frombuffer(wrap.tobytes(), wrap.shape[1::-1],"RGB")
        return wrap, imgSurface

    def customCaptureAndProcess(self) :
        # custom function to take a picture at full res
        # and convert it to cv2 RGB array
        # first, reset YUV array for next capture
        self.output = PiYUVArray(self, size=self.resolution)
        self.capture(self.output, 'yuv')
        displayText("wait while image is saving")
        self.lastframe = self.output.array.copy()
        print("frame securised")
        rgb = cv2.cvtColor(self.lastframe, cv2.COLOR_YUV2RGB)
        rgb = simplest_cb(rgb) # auto color balance function
        full = self.undistort(rgb)
        wrap, imgSurface = self.postProcess(full)

        frames.append(imgSurface)
        #
        return full, wrap

    def customCapture(self) :
        # custom function to take a picture at full res
        # and convert it to cv2 RGB array
        self.output = PiYUVArray(self, size=self.resolution)
        self.capture(self.output, 'yuv')
        return cv2.cvtColor(self.output.array, cv2.COLOR_YUV2RGB)

class MyCameraThread ():
    def __init__(self, resolution, framerate, **kwargs):
        super(MyCameraThread, self).__init__()
        # initailise camera
        # resolution / framerate / sensor_mode
        self.camera = MyCamera(resolution, framerate, 2)

    def run(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update (self):
        print("Camera thread is running")
        return self.camera


# IMAGE CONFIG  ############################################
def cornerPinTool() :
    print("CornerPin tool is active")
    # get keyboard infos
    corner = 0

    canvas.fill(BG_COLOR)
    pygame.display.flip() 
    # 
    cv2.waitKey(50)
    image = None
    image = mycamera.customCapture()
    image = mycamera.undistort(image)

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert to grayscale
    #image = exposure.rescale_intensity(image, out_range = (0, 255))
    # image warping
    #m = cv2.getPerspectiveTransform(refPoints, destPoints)
    #wraped1 = cv2.warpPerspective(image, m, (SCREEN_WIDTH, SCREEN_HEIGHT))
    wraped1, imgSurface = mycamera.postProcess(image)
    displaySurfaceImage(imgSurface)

    new_dst = np.copy(destPoints)
    wrap = image.copy()
    
    while True :
        if event.type == KEYDOWN:
            key = event.key
            if key == pygame.K_a :
                    corner = 0
            elif key == pygame.K_z :
                    corner = 1
            elif key == pygame.K_e :
                    corner = 2
            elif key == pygame.K_r :
                    corner = 3
            if key == pygame.K_UP :
                new_dst[corner][1] += 1*CORNERPIN_STEP
            elif key == pygame.K_DOWN :
                new_dst[corner][1] -= 1*CORNERPIN_STEP
            elif key == pygame.K_LEFT :
                new_dst[corner][0] -= 1*CORNERPIN_STEP
            elif key == pygame.K_RIGHT :
                new_dst[corner][0] += 1*CORNERPIN_STEP
        
            m = cv2.getPerspectiveTransform(refPoints, new_dst)
            wrapCorrected = cv2.warpPerspective(wrap, m,  (SCREEN_WIDTH, SCREEN_HEIGHT))
            imgSurface = pygame.image.frombuffer(wrapCorrected.tobytes(), wrapCorrected.shape[1::-1],"RGB")
            displaySurfaceImage(imgSurface)
            pygame.draw.circle(canvas, RED, new_dst[corner], 20)
            pygame.display.flip()

            if key == pygame.K_t:
                rfpts = refPoints.tolist() #
                dtpts = new_dst.tolist() # nested lists with same data, indices
                updateXMLDatas(rfpts, dtpts)

        if cornerPinMode == False :
            print("quit corner pin mode")
            LoadXMLDatas()
            displayLastFrame()
            break


def simplest_cb(img, percent=1):
    # from https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)
    

# XML WORK  ###################################################
def LoadXMLDatas():
    global refPoints, destPoints, camera_matrix, dist_coeffs, camResolution, camGains, camExposure
    # xml parsing
    parser = ET.XMLParser(encoding="utf-8",remove_blank_text=True)
    #fs=cv2.FileStorage(props_file,cv2.FILE_STORAGE_READ) # open cv parsing
    fs = cv2.FileStorage(XML_PATH, cv2.FILE_STORAGE_READ, encoding="UTF-8") # for openCV matrix
    tree = ET.parse(XML_PATH, parser)
    root = tree.getroot() # for standard XML parsing
    #
    refPoints = np.float32(np.array([np.array(x.split()).astype(np.float32) for x in root.findall("*/refPoints")[0].text.splitlines()]))
    destPoints = np.float32(np.array([np.array(x.split()).astype(np.float32) for x in root.findall("*/destPoints")[0].text.splitlines()]))
    # camera setup
    camResolution = np.fromstring(root.find("cameraResolution").text, dtype=int, sep=' ')
    camExposure = int(root.find("cameraSetup/shutter-speed").text)
    camGains = ()
    for x in root.find("cameraSetup/awb-gains").text.splitlines() :
        f = x.split()
        tmpFrac = frac(int(f[0]),int(f[1]))
        camGains = camGains + (tmpFrac,)
    # cv2 distortion matrix
    camera_matrix = fs.getNode("cameraMatrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()

    return root
    
def updateXMLDatas(rpts, dpts):
    global root 
    # read datas from existing XML root, and update to file
    r = root.find('distortionParam/refPoints')
    r.text = '\n'.join(map(str, [" ".join(item) for item in np.asarray(rpts).astype(str)]))
    d = root.find('distortionParam/destPoints')
    d.text = '\n'.join(map(str, [" ".join(item) for item in np.asarray(dpts).astype(str)]))
    # save updated root to file
    xml_str = ('<?xml version="1.0" encoding="UTF-8"?>\n'.encode('utf-8') + ET.tostring(root, method='xml', pretty_print=True, encoding='utf-8'))
    with open(str(XML_PATH), 'wb') as xml_file:
         xml_file.write(xml_str)

    print("[INFO] XML updated")



# SETUP  ###################################################
def setupGpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SHOT_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(18, GPIO.FALLING, callback=actionButtn)
    GPIO.setup(PLAY_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(21, GPIO.FALLING, callback=actionButtn)
    GPIO.setup(25,GPIO.OUT) # SHOT_LED


    GPIO.output(25,GPIO.HIGH)


def setupDisplay():
    global canvas, SCREEN_WIDTH, SCREEN_HEIGHT, clock
    pygame.init()
    canvas = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    #clock = pygame.time.Clock()
    infoObject = pygame.display.Info()
    print("Display resolution : %s %s" % (infoObject.current_w, infoObject.current_h))
    SCREEN_WIDTH=infoObject.current_w
    SCREEN_HEIGHT=infoObject.current_h

def loadSavedAnimation():
    global frame_counter, frames
    '''
        this function analyse working dir and check if frames already saved
        if true :
            -> populates frames array
            -> define last frame of anim to display on canvas
            -> update frame num
        else :
            display blank image

    '''
    #Get the latest image file in the working directory
    valid_files = [os.path.join(working_dir, filename) for filename in os.listdir(working_dir)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and f.rsplit('.',1)[-1] in VALID_EXT and os.path.isfile(f)]

    if not valid_files:
            # working dir is empty --> show empty frame
            displayBGColor()
            print("[INFO] No animation to load")
    else :
        #valid_files.sort() # sort files array, just to make sure last file is at the end
        valid_files = sorted(valid_files, key=os.path.getmtime, reverse = True)
        #last_file = max(valid_files, key=os.path.getmtime)
        last_file = valid_files[0]
        frame_counter = int(os.path.splitext(os.path.basename(last_file))[0])+1
        # populate frames array with valid files
        valid_files = valid_files[:maxFramesBuffer] # keep only last X frames in array
        for f in valid_files :
            image = cv2.imread(f)
            frames.appendleft(convertToSurfaceImg(image))
        # show image to screen
        canvas.fill(BG_COLOR) # to get rid of splashscreen
        displayLastTake()


def parse_args(wk):
    """
    A command line argument parser
    :return:
    """
    ap = argparse.ArgumentParser()
    # passed arguments
    ap.add_argument('-p', '--projectName', type=str, default=wk, help='project name & dir name')
    ap.add_argument('-g', '--gabarit', type=bool, default=True, help='Use gabarit ? True / False')
    ap.add_argument('-gf', '--gabaritName', type=str, default="gabarit.jpg", help='Gabarit file name')
    ap.add_argument('-d','--datas', type=str, default="cameraDatas.xml", help="file with camera datas")
    ap.add_argument('-pw','--preview', type=float, default=2, help="anim preview duration")
    #
    return vars(ap.parse_args())

#############################################################
def quit():
    camthread.camera.close()
    pygame.quit()
    sys.exit()
    GPIO.cleanup()

if __name__== "__main__":
    #global camera, timer, args, frames, XML_PATH, working_dir
    global working_dir, timer, args, XML_PATH, USE_GABARIT, gabarit, frames, mycamera, root, FONT

    now = datetime.now() # current date and time
    default_working_dir = now.strftime("images_%m_%d_%Y")
    timer = time.time()
    args = parse_args(default_working_dir)
    #
    print("\n######################  OCODA ######################\n\
    an object for collaborative animation\n######################################################")
    
    try :
        # setup function
        setupGpio()
        setupDisplay()
    except OSError:
        print ("[ERROR] Loading setup functions")
    else:
        print ("Successfully setuped user interface")

    #displayText("OCODA\na collaborative animation tool by Mamatus Collective\n\nDraw & Push the buttons!")
    # at startup, display splashscreen
    if SPLASHSCREEN_TIME > 0 :
        displaySpashScreen("resources/splashcreen.jpg")
        cv2.waitKey(SPLASHSCREEN_TIME)


    working_dir = os.getcwd() + "/_projects/" + args.get('projectName') # we add a "_" to have working dirs always on top in dir
    isDir = os.path.isdir(working_dir)
    if isDir == False :
        try:
            os.mkdir(working_dir);
        except OSError:
            print ("Creation of the directory %s failed" % working_dir)
            quit()
        else:
            print ("Working directory : %s" % working_dir)
    else :
        print ("Working directory : %s" % working_dir)

    archiveDir = working_dir+"/full-res-img/"
    isDir = os.path.isdir(archiveDir)
    if isDir == False :
        try:
            os.mkdir(archiveDir);
        except OSError:
            print ("Creation of the directory %s failed" % archiveDir)
            quit()
        else:
            print ("Image archiving directory : %s" % archiveDir)
    else :
        print ("Image archiving directory : %s" % archiveDir)

    # undistort datas loading
    root = None
    try:
        # setup camera datas xml path
        xml = args.get('datas')
        XML_PATH = working_dir+"/"+xml
        # if xml not in working dir --> copy from original
        while not os.path.isfile(XML_PATH) :
            print("copy cam datas to working dir")
            shutil.copy2(xml, working_dir)

        print("Loading XML Camera Datas : %s" %XML_PATH)
        root = LoadXMLDatas()

    except OSError:
        print ("[ERROR] Loading of camera datas has failed !")
        print ("--> please make camera calibration before starting")
    else:
        print ("Successfully loaded camera datas")
    
    
    try :
        USE_GABARIT = args.get('gabarit')
        if USE_GABARIT :
            gabFileName = str(args.get('gabaritName'))
            gabpath = working_dir + "/" + gabFileName

            if not os.path.isfile(gabpath) :
                # no gabarit file in working dir, create one from template
                shutil.copy2("resources/gabarit.jpg", gabpath) # resources/

            gabarit = pygame.image.load(gabpath)
            gabarit = pygame.transform.scale(gabarit, (SCREEN_WIDTH,SCREEN_HEIGHT))
            gabarit.convert_alpha()

    except OSError:
        print ("[ERROR] Loading gabarit file '%s' failed" %gabpath)
    else:
        print ("Successfully loaded gabarit file '%s'" %gabpath)

    # frames buffer --> ring buffer
    PREVIEW_DURATION = args.get('preview') # duration in seconds for animation preview (last X seconds)
    maxFramesBuffer = int(PREVIEW_DURATION*FPS)
    print(maxFramesBuffer)
    frames = collections.deque(maxlen=maxFramesBuffer)
    print("Anim preview duration : {} sec @ {}fps".format(PREVIEW_DURATION,FPS))
    #
    #canvas.fill(BG_COLOR)
    #pygame.display.flip()
    # start camera thread
    camthread = MyCameraThread(camResolution, FPS).run()
    mycamera = camthread.camera
    print("Camera resolution : %s" %np.array(mycamera.resolution))

    # activity analyse
    activityCheck() # timer thread that analyse activity every X seconds
    if USE_MOTIONDETECTION :
        # hazardous // do no use until it is cleaned !
        motionDetection = Thread(target=detectMotion, args=(), daemon = True)
        motionDetection.start()
        print("Motion detection used -- ! hasardous !")
    
    leds = Thread(target=blinkLed, daemon=True)
    leds.start()
    # we can start animation now !
    loadSavedAnimation()
    #
    print("Ready to animate !")
    #
    FONT = pygame.font.SysFont("comicsansms", 72)
    #make_popup()
    while True:
        #blinkLed()
        for event in pygame.event.get():
            if event.type == QUIT:
                quit()
            # https://www.pygame.org/docs/ref/key.html
            elif event.type == KEYDOWN:
                print("Key pressed : {}\t{}\t{}".format(event.key, event.scancode, event.unicode))
                updateTimer ()
                key = event.key
                if key == pygame.K_c :
                    # corner pin mode
                    cornerPinMode = not cornerPinMode
                    if cornerPinMode == True :
                        print("Enter corner pin mode")
                        cornerPin = Thread(target=cornerPinTool, args=(), daemon = True)
                        cornerPin.start()
                    else :
                        LoadXMLDatas() # reload XML datas after cornerpin correction
                        displayLastFrame()
                elif key == pygame.K_s:
                    actionButtn(SHOT_BUTTON)
                elif key == pygame.K_p:
                    if not(is_playing):
                        actionButtn(PLAY_BUTTON)
                elif key == pygame.K_f:
                    displayVideoStream ()

                elif key == pygame.K_q or key == pygame.K_ESCAPE or event.scancode == 180:
                    print("quit")
                    quit()
                    break


# RESSOURCES  ###################################################
#fisheye calibration
#https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# GPIO Input debounce
# https://www.raspberrypi.org/forums/viewtopic.php?f=28&t=131440&sid=4a76b513556f04284360808e2dc88bda&start=52
