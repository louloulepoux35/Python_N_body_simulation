import numpy as np
import math


class Particule:
    def __init__(self, px, py, pmass, vx, vy):
        self.pos = np.array([px, py])
        self.prevPos = self.pos
        self.velocity = np.array([vx, vy])
        self.mass = pmass
        self.quadTreeNode = None

class QuadTreeNode:
    def __init__(self, px0, py0, px1, py1, pparent, pchildren=[]):
        self.x0 = px0
        self.y0 = py0
        self.x1 = px1
        self.y1 = py1
        self.children = pchildren
        self.parent = pparent
        self.particuleIndex = -1
        self.centerOfGravity = np.array([0, 0])
        self.mass = 0
        self.depth = 0
        if pparent != None:
            self.depth = pparent.depth + 1

    def drawQuadNode(self, drawBuffer):
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



    def create4Children(self):
        if len(self.children) == 0:
            self.children.append(QuadTreeNode(self.x0, self.y0, (self.x0+self.x1) / 2, (self.y0+self.y1) / 2, self, []))
            self.children.append(QuadTreeNode((self.x0 + self.x1) / 2, self.y0, self.x1, (self.y0 + self.y1) / 2, self, []))
            self.children.append(QuadTreeNode(self.x0, (self.y0 + self.y1) / 2, (self.x0+self.x1) / 2, self.y1, self, []))
            self.children.append(QuadTreeNode((self.x0+self.x1) / 2, (self.y0 + self.y1) / 2, self.x1, self.y1, self, []))



class simulator():
    def __init__(self, centerX, centerY, radius, resolution, scale, galaxyMass, gConst, friction, initialVelocity, theta):
        # Init particules
        self.particules = np.array([])

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


    def computeAverageMass(self, quadTreeNode):
        if len(quadTreeNode.children) == 0:
            if quadTreeNode.particuleIndex != -1:
                quadTreeNode.mass = self.particules[quadTreeNode.particuleIndex].mass
                quadTreeNode.centerOfGravity = self.particules[quadTreeNode.particuleIndex].pos
            else:
                quadTreeNode.mass = 0
        else:
            avgMass = 0
            avgGravityCenter = np.array([0, 0])
            for node in quadTreeNode.children:
                self.computeAverageMass(node)
                avgMass += node.mass
                avgGravityCenter = avgGravityCenter + (node.centerOfGravity * node.mass)
            quadTreeNode.centerOfGravity = avgGravityCenter / avgMass
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


    def twoParticlesForces(self, thisPos, pos2, thisMass, mass2, sommeForces):
        vect = pos2 - thisPos
        dist = np.linalg.norm(vect)
        distClamped = dist
        if distClamped < 20:
            distClamped = 20
        vDir = vect / distClamped
        gForce = self.gConst * (thisMass * mass2) / (distClamped * distClamped * distClamped)

        # if dist < 20 and dist > 0:
        # gForce = (-1/dist) * self.gConst * (particle.mass * child.mass) * 0.1
        sommeForces = sommeForces + (gForce * vDir)
        return sommeForces


    def isNodeFarEnought(self, particle, node):
        nodeSizeX = math.fabs(node.x1 - node.x0)
        nodeSizeY = math.fabs(node.y1 - node.y0)
        nodeSize = (nodeSizeX + nodeSizeY) / 2
        vect = particle.pos - node.centerOfGravity
        dist = np.linalg.norm(vect)
        if dist > 0:
            theta = nodeSize / dist
            if theta < self.theta:
                return True
        return False


    def rewindTreeComputeForces(self, particle, sommeForces, node, nodeFrom):
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


    def parcourSubTree(self, node, particle, sommeForces):
        if len(node.children) > 0 and self.isNodeFarEnought(particle, node) == False:
            for child in node.children:
                sommeForces = self.parcourSubTree(child, particle, sommeForces)
        else:
            sommeForces = self.twoParticlesForces(particle.prevPos, node.centerOfGravity, particle.mass, node.mass, sommeForces)

        return sommeForces


    def pfd(self, sommeForces, partIndex):
        frictionForce = self.particules[partIndex].velocity * self.friction
        sommeForces -= frictionForce
        accel = sommeForces / self.particules[partIndex].mass
        self.particules[partIndex].velocity = self.particules[partIndex].velocity + accel
        self.particules[partIndex].pos = self.particules[partIndex].pos + self.particules[partIndex].velocity


    def updateParticules(self, quadTreeRoot):
        # Update
        nbPart = len(self.particules)
        for i in range(nbPart):
           # Update particule
           sommeForces = np.array([0, 0])
           sommeForces = self.rewindTreeComputeForces(self.particules[i], sommeForces, self.particules[i].quadTreeNode.parent, self.particules[i].quadTreeNode)
           self.pfd(sommeForces, i)

        # if len(quadTreeRoot.children) > 0:
        #     for child in quadTreeRoot.children:
        #         self.updateParticules(child)
        # else:
        #     if quadTreeRoot.particuleIndex != -1:
        #         # Update particule
        #         sommeForces = np.array([0, 0])
        #         sommeForces = self.rewindTreeComputeForces(self.particules[quadTreeRoot.particuleIndex], sommeForces, quadTreeRoot.parent, quadTreeRoot)
        #         self.pfd(sommeForces, quadTreeRoot.particuleIndex)