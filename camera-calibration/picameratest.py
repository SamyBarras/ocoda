import picamera
import time
import pygame, sys
from pygame.locals import * 


pygame.init()
#canvas = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
canvas = pygame.display.set_mode((0,0),pygame.RESIZABLE)

with picamera.PiCamera() as camera:
  camera.sensor_mode=2
  camera.resolution = (2592, 1944)
  camera.start_preview()
  while True:
    key = False
    for event in pygame.event.get():
      if event.type == KEYDOWN:
        key = event.key
      if key == pygame.K_q:
        camera.stop_preview()
        camera.close()
        pygame.quit()
        sys.exit()
        break
