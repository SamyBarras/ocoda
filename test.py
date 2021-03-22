#! /usr/bin/env python3
import picamera
import pygame
import io

# Init pygame 
pygame.init()
screen = pygame.display.set_mode((500, 500), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)

# Init camera
camera = picamera.PiCamera()
#camera.sensor_mode=0
camera.resolution = (2592, 1944)
#camera.resolution = (1296, 972)
#camera.crop = (0.0, 0.0, 1.0, 1.0)

x = (screen.get_width() - camera.resolution[0]) / 2
y = (screen.get_height() - camera.resolution[1]) / 2

# Init buffer
rgb = bytearray(camera.resolution[0] * camera.resolution[1] * 3)

# Main loop
exitFlag = True
while(exitFlag):

    stream = io.BytesIO()
    camera.capture(stream, use_video_port=True, format='rgb')
    stream.seek(0)
    stream.readinto(rgb)
    stream.close()
    img = pygame.image.frombuffer(rgb[0:
          (camera.resolution[0] * camera.resolution[1] * 3)],
           camera.resolution, 'RGB')

    screen.fill(0)
    if img :
        screen.blit(pygame.transform.scale(img, (200, 200)), (0, 0))


    for event in pygame.event.get():
        if(event.type is pygame.MOUSEBUTTONDOWN or 
           event.type is pygame.QUIT):
            exitFlag = False
        elif (event.type is pygame.KEYDOWN and event.key == pygame.K_i and img):
            #screen.blit(img, (x,y))
            screen.blit(img, (0,0))
        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(event.dict['size'], pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
            screen.blit(pygame.transform.scale(img, event.dict['size']), (0, 0))
            pygame.display.flip()

    pygame.display.update()

camera.close()
pygame.display.quit()