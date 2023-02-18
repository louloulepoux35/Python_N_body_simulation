# Build with: python setup.py build_ext --inplace

import numpy as np
cimport numpy as np
import math
from libc.math cimport sqrt
from libc.math cimport fabs
#from cython.parallel import prange


cdef class Particule:
    cdef public np.float64_t[:] pos
    cdef public np.float64_t[:] prevPos
    cdef public np.float64_t[:] velocity
    cdef public double mass
    cdef public QuadTreeNode quadTreeNode

    def __cinit__(self, double px, double py, double pmass, double vx, double vy):
        self.pos = np.array([px, py])
        self.prevPos = self.pos
        self.velocity = np.array([vx, vy])
        self.mass = pmass
        self.quadTreeNode = None


cdef class QuadTreeNode:
    cdef public double x0, y0, x1, y1, mass
    cdef public np.float64_t[:] centerOfGravity
    cdef public int particuleIndex, depth
    cdef public QuadTreeNode parent
    cdef public list children

    def __cinit__(self, double px0, double py0, double px1, double py1, QuadTreeNode pparent, list pchildren=[]):
        self.x0 = px0
        self.y0 = py0
        self.x1 = px1
        self.y1 = py1
        self.children = pchildren
        self.parent = pparent
        self.particuleIndex = -1
        self.centerOfGravity = np.array([0.0, 0.0])
        self.mass = 0.0
        self.depth = 0
        if pparent != None:
            self.depth = pparent.depth + 1

    cpdef drawQuadNode(self, drawBuffer):
        drawBuffer.line((self.x0, self.y0, self.x1, self.y0), fill="green")
        drawBuffer.line((self.x1, self.y0, self.x1, self.y1), fill="green")
        drawBuffer.line((self.x1, self.y1, self.x0, self.y1), fill="green")
        drawBuffer.line((self.x0, self.y1, self.x0, self.y0), fill="green")

        if self.mass > 0:
            drawBuffer.point((self.centerOfGravity[0], self.centerOfGravity[1]), fill="yellow")
            #dotSize = self.mass / GALAXY_MASS
            #drawBuffer.ellipse((self.centerOfGravity[0], self.centerOfGravity[1], self.centerOfGravity[0] + dotSize, self.centerOfGravity[1] + dotSize), fill="yellow")

        if len(self.children) > 0:
            for node in self.children:
                node.drawQuadNode(drawBuffer)



    cpdef create4Children(self):
        if len(self.children) == 0:
            self.children.append(QuadTreeNode(self.x0, self.y0, (self.x0+self.x1) / 2, (self.y0+self.y1) / 2, self, []))
            self.children.append(QuadTreeNode((self.x0 + self.x1) / 2, self.y0, self.x1, (self.y0 + self.y1) / 2, self, []))
            self.children.append(QuadTreeNode(self.x0, (self.y0 + self.y1) / 2, (self.x0+self.x1) / 2, self.y1, self, []))
            self.children.append(QuadTreeNode((self.x0+self.x1) / 2, (self.y0 + self.y1) / 2, self.x1, self.y1, self, []))



cdef class simulator():
    cdef public np.ndarray particules

    cdef public double galaxyMass
    cdef public double gConst
    cdef public double friction
    cdef public double initialVelocity
    cdef public double theta

    def __cinit__(self, double centerX, double centerY, double radius, int resolution, double scale, double galaxyMass, double gConst, double friction, double initialVelocity, double theta):
        # Init particules
        self.particules = np.array([], dtype=object)

        self.galaxyMass = galaxyMass
        self.gConst = gConst
        self.friction = friction
        self.initialVelocity = initialVelocity
        self.theta = theta

        self.initParticles(resolution, scale, centerX, centerY, radius)


    def initParticles(self, resolution, multiple, centerX, centerY, radius):
        maxSize = resolution * multiple
        for x in range(resolution):
            x = x * multiple
            for y in range(resolution):
                y = y * multiple
                pos = np.array([float(x) + centerX - maxSize / 2, float(y) + centerY - maxSize / 2])
                vect = pos - np.array([centerX, centerY])
                dist = np.linalg.norm(vect)
                #rnd = rand.randint(0, 10000)
                #if rnd > 9800 and dist < radius:
                if dist < radius:
                    velocityFactor = dist / radius
                    c = np.array([centerX, centerY])
                    v1 = pos - c

                    # Bad luck, particle right in the middle, skip it to avoid to divide by 0...
                    if np.linalg.norm(v1) == 0:
                        continue

                    v1 = v1 / np.linalg.norm(v1)
                    tan = np.array([v1[1], -v1[0]])
                    tan = tan * self.initialVelocity * velocityFactor
                    self.particules = np.append(self.particules, Particule(pos[0], pos[1], self.galaxyMass, tan[0], tan[1]))
                    # particules.append(Particule(pos[0], pos[1], self.galaxyMass, 0, 0))
                    nbTmp = 0

        print(f'Nb particules:, {len(self.particules)}')


    cpdef computeAverageMass(self, QuadTreeNode quadTreeNode):
        cdef double avgMass = 0.0
        cdef np.ndarray[np.float64_t, ndim=1] avgGravityCenter = np.zeros(2)
        if len(quadTreeNode.children) == 0:
            if quadTreeNode.particuleIndex != -1:
                quadTreeNode.mass = self.particules[quadTreeNode.particuleIndex].mass
                quadTreeNode.centerOfGravity = self.particules[quadTreeNode.particuleIndex].pos
            else:
                quadTreeNode.mass = 0.0
        else:
            avgMass = 0.0
            avgGravityCenter = np.array([0.0, 0.0])
            for node in quadTreeNode.children:
                self.computeAverageMass(node)
                avgMass += node.mass
                #avgGravityCenter = avgGravityCenter + (node.centerOfGravity * node.mass) // TODO: XXX ??
                avgGravityCenter[0] = avgGravityCenter[0] + (node.centerOfGravity[0] * node.mass)
                avgGravityCenter[1] = avgGravityCenter[1] + (node.centerOfGravity[1] * node.mass)
            quadTreeNode.centerOfGravity[0] = avgGravityCenter[0] / avgMass
            quadTreeNode.centerOfGravity[1] = avgGravityCenter[1] / avgMass
            quadTreeNode.mass = avgMass


    def insertParticle(self, quadTreeNode, pIndex):
        if self.particules[pIndex].pos[0] >= quadTreeNode.x0 and self.particules[pIndex].pos[0] <= quadTreeNode.x1:
            if self.particules[pIndex].pos[1] >= quadTreeNode.y0 and self.particules[pIndex].pos[1] <= quadTreeNode.y1:
                if len(quadTreeNode.children) == 0 and quadTreeNode.particuleIndex == -1:
                    quadTreeNode.particuleIndex = pIndex
                    self.particules[pIndex].quadTreeNode = quadTreeNode
                    return
                if len(quadTreeNode.children) != 0:
                    for child in quadTreeNode.children:
                        self.insertParticle(child, pIndex)
                    return
                if len(quadTreeNode.children) == 0 and quadTreeNode.particuleIndex != -1:
                    quadTreeNode.create4Children()
                    for child in quadTreeNode.children:
                        self.insertParticle(child, pIndex)
                    for child in quadTreeNode.children:
                        self.insertParticle(child, quadTreeNode.particuleIndex)
                    quadTreeNode.particuleIndex = -1



    def createQuadTree(self):
        newQuadTree = QuadTreeNode(0, 0, 0, 0, None, [])
        x0 = 0
        y0 = 0
        x1 = 0
        y1 = 0
        for i in range(len(self.particules)):
            if i == 0:
                x0 = self.particules[i].pos[0]
                y0 = self.particules[i].pos[1]
                x1 = x0
                y1 = y0
            if self.particules[i].pos[0] < x0:
                x0 = self.particules[i].pos[0]
            if self.particules[i].pos[1] < y0:
                y0 = self.particules[i].pos[1]
            if self.particules[i].pos[0] > x1:
                x1 = self.particules[i].pos[0]
            if self.particules[i].pos[1] > y1:
                y1 = self.particules[i].pos[1]

        newQuadTree.x0 = x0
        newQuadTree.x1 = x1
        newQuadTree.y0 = y0
        newQuadTree.y1 = y1

        for i in range(len(self.particules)):
            self.insertParticle(newQuadTree, i)

        return newQuadTree


    cdef np.ndarray[np.float64_t, ndim=1] twoParticlesForces(self, np.ndarray[np.float64_t, ndim=1] thisPos, np.ndarray[np.float64_t, ndim=1] pos2, double thisMass, double mass2, np.ndarray[np.float64_t, ndim=1] sommeForces):
        cdef np.ndarray[np.float64_t, ndim=1] vect = np.zeros(2)
        vect[0] = pos2[0] - thisPos[0]
        vect[1] = pos2[1] - thisPos[1]
        #dist = np.linalg.norm(vect)
        cdef double dist = sqrt((pos2[0] - thisPos[0])**2 + (pos2[1] - thisPos[1])**2)
        cdef double distClamped = dist if dist >= 20 else 20

        cdef np.ndarray[np.float64_t, ndim=1] vDir = np.zeros(2)
        vDir[0] = (pos2[0] - thisPos[0]) / distClamped
        vDir[1] = (pos2[1] - thisPos[1]) / distClamped

        cdef double gForce = self.gConst * (thisMass * mass2) / (distClamped * distClamped * distClamped)

        # if dist < 20 and dist > 0:
        # gForce = (-1/dist) * self.gConst * (particle.mass * child.mass) * 0.1
        sommeForces[0] = sommeForces[0] + (gForce * vDir[0])
        sommeForces[1] = sommeForces[1] + (gForce * vDir[1])

        return sommeForces


    cdef bint isNodeFarEnought(self, Particule particle, QuadTreeNode node):
        cdef double nodeSizeX = fabs(node.x1 - node.x0)
        cdef double nodeSizeY = fabs(node.y1 - node.y0)
        cdef double nodeSize = (nodeSizeX + nodeSizeY) / 2
        cdef np.ndarray[np.float64_t, ndim=1] vect = np.zeros(2)
        vect[0] = particle.pos[0] - node.centerOfGravity[0]
        vect[1] = particle.pos[1] - node.centerOfGravity[1]
        #cdef double dist = np.linalg.norm(vect)
        cdef double dist = sqrt(vect[0]**2 + vect[1]**2)
        cdef double theta
        if dist > 0:
            theta = nodeSize / dist
            if theta < self.theta:
                return True
        return False


    cdef np.ndarray[np.float64_t, ndim=1] rewindTreeComputeForces(self, Particule particle, np.ndarray[np.float64_t, ndim=1] sommeForces, QuadTreeNode node, QuadTreeNode nodeFrom):
        if node == None:
            return sommeForces
        for child in node.children:
            # Don't check the node we come from
            if child == nodeFrom:
                continue
            if child.mass > 0:
                # If the node are far enough, compute with approximation
                # If the node are too close, go further down on the tree
                sommeForces = self.parcourSubTree(child, particle, sommeForces)

                # Huge approximation, lead to visible weird effect
                #sommeForces = twoParticlesForces(particle.prevPos, child.centerOfGravity, particle.mass, child.mass, sommeForces)

        return self.rewindTreeComputeForces(particle, sommeForces, node.parent, node)


    cdef np.ndarray[np.float64_t, ndim=1] parcourSubTree(self, QuadTreeNode node, Particule particle, np.ndarray[np.float64_t, ndim=1] sommeForces):
        cdef int num_children = len(node.children)
        cdef QuadTreeNode child
        cdef np.ndarray[np.float64_t, ndim=1] tmp1 = np.zeros(2)
        cdef np.ndarray[np.float64_t, ndim=1] tmp2 = np.zeros(2)
        if num_children > 0 and self.isNodeFarEnought(particle, node) == False:
            for i in range(num_children):
                child = node.children[i]
                sommeForces = self.parcourSubTree(child, particle, sommeForces)
        else:
            tmp1[0] = particle.prevPos[0] # TODO XXX WTF ??
            tmp1[1] = particle.prevPos[1]
            tmp2[0] = node.centerOfGravity[0]
            tmp2[1] = node.centerOfGravity[1]
            sommeForces = self.twoParticlesForces(tmp1, tmp2, particle.mass, node.mass, sommeForces)
            #sommeForces = self.twoParticlesForces(particle.prevPos, node.centerOfGravity, particle.mass, node.mass, sommeForces)

        return sommeForces


    cdef void pfd(self, np.ndarray[np.float64_t, ndim=1] sommeForces, int partIndex):
        cdef np.ndarray[np.float64_t, ndim=1] frictionForce = np.zeros(2, dtype=np.float64)
        frictionForce[0] = self.particules[partIndex].velocity[0] * self.friction
        frictionForce[1] = self.particules[partIndex].velocity[1] * self.friction
        sommeForces[0] -= frictionForce[0]
        sommeForces[1] -= frictionForce[1]
        cdef np.ndarray[np.float64_t, ndim=1] accel = np.zeros(2, dtype=np.float64)
        accel[0] = sommeForces[0] / self.particules[partIndex].mass
        accel[1] = sommeForces[1] / self.particules[partIndex].mass
        self.particules[partIndex].velocity[0] = self.particules[partIndex].velocity[0] + accel[0]
        self.particules[partIndex].velocity[1] = self.particules[partIndex].velocity[1] + accel[1]
        self.particules[partIndex].pos[0] = self.particules[partIndex].pos[0] + self.particules[partIndex].velocity[0]
        self.particules[partIndex].pos[1] = self.particules[partIndex].pos[1] + self.particules[partIndex].velocity[1]


    cdef oN2(self):
        # Update
        cdef np.ndarray[np.float64_t, ndim=1] sommeForces = np.zeros(2, dtype=np.float64)
        cdef int N = len(self.particules)
        cdef int i, j = 0
        cdef np.ndarray[np.float64_t, ndim=1] tmp1 = np.zeros(2)
        cdef np.ndarray[np.float64_t, ndim=1] tmp2 = np.zeros(2)
        for i in range(N):
            sommeForces = np.zeros(2, dtype=np.float64)
            for j in range(N):
                if i != j:
                    tmp1[0] = self.particules[i].prevPos[0] # TODO XXX WTF ??
                    tmp1[1] = self.particules[i].prevPos[1]
                    tmp2[0] = self.particules[j].prevPos[0]
                    tmp2[1] = self.particules[j].prevPos[1]
                    sommeForces = self.twoParticlesForces(tmp1, tmp2, self.particules[i].mass, self.particules[j].mass, sommeForces)
            self.pfd(sommeForces, i)


    cpdef updateParticules(self, QuadTreeNode quadTreeRoot):
        # Update
        #self.oN2()


        cdef int nbPart = len(self.particules)
        cdef np.ndarray[np.float64_t, ndim=1] sommeForces = np.zeros(2, dtype=np.float64)
        cdef int i = 0
        #for i in prange(nbPart, nogil=True):
        for i in range(nbPart):
           # Update particule
           sommeForces = np.zeros(2, dtype=np.float64)
           sommeForces = self.rewindTreeComputeForces(self.particules[i], sommeForces, self.particules[i].quadTreeNode.parent, self.particules[i].quadTreeNode)
           self.pfd(sommeForces, i)





        #cdef np.ndarray[np.float64_t, ndim=1] sommeForces = np.zeros(2, dtype=np.float64)
        #cdef int i, N = len(quadTreeRoot.children)
        #if N > 0:
        #    for i in range(N):
        #        self.updateParticules(quadTreeRoot.children[i])
        #else:
        #    if quadTreeRoot.particuleIndex != -1:
        #        # Update particule
        #        sommeForces = self.rewindTreeComputeForces(self.particules[quadTreeRoot.particuleIndex], sommeForces, quadTreeRoot.parent, quadTreeRoot)
        #        self.pfd(sommeForces, quadTreeRoot.particuleIndex)
