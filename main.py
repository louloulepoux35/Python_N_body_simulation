# https://anaroxanapop.github.io/behalf/

# Improvments:
# If particles are too close, divide the dt so the error stay acceptable
# If particles are far away, increase dt to save compute power

import random
import time
from PIL import Image, ImageDraw
from random import Random
import numpy as np
import cv2
import math
import sys
import threading
from multiprocessing import Process
#from computeSlow import simulator
from computeFast import simulator


# Init randomness
#rand = random.SystemRandom()

# Circle
#centerX = 400
#centerY = 400
#radius = 150
#resolution = 25
#for alpha in range(360):
 #   px = centerX + radius * math.cos(alpha)
 #   py = centerY + radius * math.sin(alpha)
 #   p = np.array([px, py])
 #   c = np.array([centerX, centerY])
 #   v1 = p - c
 #   v1 = v1 / np.linalg.norm(v1)
 #   tan = np.array([v1[1], -v1[0]])
 #   tan = tan * INITIAL_VELOCITY
 #   particules.append(Particule(float(px), float(py), GALAXY_MASS, tan[0], tan[1]))

#for x in range(resolution):
 #   for y in range(resolution):
 #       pos = np.array([float(x) + centerX - resolution/2, float(y) + centerY - resolution/2])
 #       vect = pos - np.array([centerX, centerY])
 #       dist = np.linalg.norm(vect)
 #       rnd = rand.randint(0, 10000)
 #       if rnd > 9800 and dist < radius:
 #           velocityFactor = dist / radius
 #           c = np.array([centerX, centerY])
 #           v1 = pos - c
  #          v1 = v1 / np.linalg.norm(v1)
  #          tan = np.array([v1[1], -v1[0]])
  #          tan = tan * INITIAL_VELOCITY * velocityFactor
  #          particules = np.append(particules, Particule(pos[0], pos[1], GALAXY_MASS, tan[0], tan[1]))
  #          #particules.append(Particule(pos[0], pos[1], GALAXY_MASS, 0, 0))
  #          nbTmp = 0






screenSize = (2048, 2048)

# Choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter('video.mp4', fourcc, 24, screenSize)


# Init simulator parameters
galaxyMass = 8000
gConst = 0.01
friction = 0
initialVelocity = 2
theta = 0.5

# Init particules numbers and positions
centerX = 1024
centerY = 1024
radius = 200
resolution = 20
scale = 20
simu = simulator(centerX, centerY, radius, resolution, scale, galaxyMass, gConst, friction, initialVelocity, theta)


frameCounter = 0

# Numbers frames of the video
NB_FRAMES = 800

start = time.time()
while True:

    drawStart = time.time()
    # Draw particules
    im = Image.new('RGB', screenSize, color='black')
    draw = ImageDraw.Draw(im)
    for part in simu.particules:
        densityColor = (255, 0, 0)
        dotSize = 1
        draw.ellipse((int(part.pos[0]), int(part.pos[1]), int(part.pos[0])+dotSize, int(part.pos[1])+dotSize), fill=densityColor)
        #draw.point((int(part.pos[0]), int(part.pos[1])), fill="red")
        # Copy pos in prevPos
        part.prevPos = part.pos
    drawEnd = time.time()
    drawTime = drawEnd - drawStart

    startTimeAlgo = time.time()

    # o(Nlog(N)) method
    node = simu.createQuadTree()
    createQuadTreeTime = time.time()
    simu.computeAverageMass(node)
    averagemassTime = time.time()
    # Uncomment this to draw the quadtree
    #node.drawQuadNode(draw)
    simu.updateParticules(node)

    endTimeAlgo = time.time()

    createTreeTime = createQuadTreeTime - startTimeAlgo
    averageMassTime = averagemassTime - createQuadTreeTime
    updateParticlesTime = endTimeAlgo - averagemassTime
    totalTime = endTimeAlgo - drawStart

    #print("Draw: ", drawTime)
    #print("Create tree: ", createTreeTime)
    #print("Avrg mass: ", averageMassTime)
    #print("Update: ", updateParticlesTime)
    #print("total time: ", totalTime)
    #print("----------------------------------")

    # o(N2) methode
    #for i in range(len(particules)):
        #sommeForces = np.array([0, 0])
        #for j in range(len(particules)):
            #if i != j:
                #sommeForces = twoParticlesForces(particules[i].prevPos, particules[j].prevPos, particules[i].mass, particules[j].mass, sommeForces)
                #pfd(sommeForces, i)
    #time.sleep(1)


    # convert image to numpy array
    frameBuffer = np.array(im)
    # Convert RGB to BGR
    frameBuffer = frameBuffer[:, :, ::-1].copy()
    cv2.imshow('Univers expansion', frameBuffer)
    video.write(frameBuffer)

    frameCounter += 1
    print(frameCounter)

    cv2.waitKey(1)

    end = time.time()
    elapsedTime = end - start
    remainingTime = (elapsedTime / frameCounter) * (NB_FRAMES - frameCounter)
    print("Remaining time (minutes): ", (remainingTime/60))
    #print("Algo dt = ", (endTimeAlgo - startTimeAlgo))

    if frameCounter >= NB_FRAMES:
        video.release()
        cv2.destroyAllWindows()
        break

