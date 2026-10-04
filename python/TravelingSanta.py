import math
import numpy as np
import csv
import os
import sys
import multiprocessing
import random
from timeit import default_timer as timer
from datetime import datetime
import pickle


R = 6378.0 # kilometers


class Point:

    def __init__(self, id, latitude, longitude, weight):
        latitude = math.radians(latitude)
        longitude = math.radians(longitude)
        x = R * math.cos(latitude) * math.cos(longitude)
        y = R * math.cos(latitude) * math.sin(longitude)
        z = R * math.sin(latitude)
        self.vector = np.array([x, y, z])
        self.id = id
        self.weight = weight
        self.index = id-1

KORVATUNTURI = Point(1, 68.073611, 29.315278, 0)

def calculateAllDistances(points):
    p1 = np.array([KORVATUNTURI.vector] + list(map(lambda p: p.vector, points)))
    p2 = np.transpose(p1)
    v = np.dot(p1, p2) / R ** 2
    v[v > 1] = 1 # no invalid values because of numeric errors
    return R*np.arccos(v)

class GiftRun:

    def __init__(self):
        self.current = []
        self.all = []
        self.totalDistance = 0

    def append(self, point, distance):
        self.current.append(point.id)
        self.totalDistance += distance

    def runFinished(self, distance):
        self.all.append(self.current.copy())
        self.current.clear()
        self.totalDistance += distance

    def __str__(self):
        me = ""
        for run in self.all:
            isFirst = True
            for id in run:
                if(not isFirst):
                    me += '; '
                me += str(id)
                isFirst = False
            me += '\n'
        return me

    @staticmethod
    def Parse(lines, totalDistance):
        me = GiftRun()
        me.totalDistance = totalDistance
        for line in lines:
            me.all.append(list(map(lambda x: float(x), line.split('; '))))
        return me

class Ant():

    def __init__(self, points: list, capacity, allDistances):

        self.pointsLeft = points.copy()
        self.possiblePoints = points.copy()
        self.capacity = capacity
        self.capacityLeft = capacity
        self.allDistances = allDistances

    def findSolution(self, queue, attractiveness, pheromone):
        dt = datetime.now()
        np.random.seed(dt.microsecond)
        lastPoint = KORVATUNTURI
        giftRun = GiftRun()
        while len(self.pointsLeft) > 0:
            if len(self.possiblePoints) > 0:
                nextPoint = self.__chooseNext(lastPoint, attractiveness, pheromone)
                if(nextPoint != KORVATUNTURI):
                    giftRun.append(nextPoint, self.allDistances[lastPoint.index, nextPoint.index])
                    self.pointsLeft.remove(nextPoint)
                    lastPoint = nextPoint
                    self.possiblePoints.remove(nextPoint)
                    self.capacityLeft -= nextPoint.weight
                    self.possiblePoints = list(filter(lambda p: p.weight <= self.capacityLeft, self.possiblePoints))
            if len(self.possiblePoints) == 0 or nextPoint == KORVATUNTURI:
                giftRun.runFinished(self.allDistances[lastPoint.index, KORVATUNTURI.index])
                self.capacityLeft = self.capacity
                self.possiblePoints = self.pointsLeft.copy()
                lastPoint = KORVATUNTURI
        giftRun.runFinished(self.allDistances[lastPoint.index, KORVATUNTURI.index])
        queue.put(giftRun)

    def __chooseNextACO(self, lastPoint, attractiveness, pheromone):
        possible = np.array(self.possiblePoints)
        #if(lastPoint != KORVATUNTURI):
        #   np.append(possible, [KORVATUNTURI])
        visit = np.zeros(possible.shape)
        for i in range(len(possible)):
            nextIndex = possible[i].index
            visit[i] = attractiveness[lastPoint.index][nextIndex]*math.pow(pheromone[lastPoint.index][nextIndex], 2)
        probabilities = visit/np.sum(visit)
        return np.random.choice(possible, 1, p=probabilities)[0]

    def __chooseNext(self, lastPoint, attractiveness, pheromone):
        if lastPoint == KORVATUNTURI:
            return random.choice(self.possiblePoints)
        if random.random() < 0.95:
            return self.__getClosest(lastPoint)

        maxP = -1
        maxPoint = self.possiblePoints[0]
        for i in range(len(self.possiblePoints)):
            nextPoint = self.possiblePoints[i]
            P = pheromone[lastPoint.index][nextPoint.index]
            if P > maxP:
                maxPoint = nextPoint
                maxP = P
        return maxPoint

    # delete
    def __getClosest(self, lastPoint):
        nextPoint = self.possiblePoints[0]
        closestDistance = sys.float_info.max
        for point in self.possiblePoints:
            d = self.allDistances[lastPoint.index, point.index]
            if d < closestDistance:
                nextPoint = point
                closestDistance = d
        return nextPoint

    # delete
    def __getWeighest(self):
        nextPoint = self.possiblePoints[0]
        weighest = 0
        for point in self.possiblePoints:
            if point.weight > weighest:
                nextPoint = point
                weighest = point.weight
        return nextPoint

def readPoints():
    filename = "nicelist.txt"
    points = []
    with open(filename, newline='') as csvfile:
        pointreader = csv.reader(csvfile, delimiter=';')
        for row in pointreader:
            id = int(row[0])
            latitude = float(row[1])
            longitude = float(row[2])
            weight = float(row[3])
            points.append(Point(id, latitude, longitude, weight))
    return points

def saveBest(giftRun):
    os.makedirs("results", exist_ok=True)
    fileName = "results/"+str(giftRun.totalDistance) +".txt"
    file = open(fileName, "w+")
    file.write(str(giftRun))


def loadBest():
    best = ""
    bestDistance = sys.float_info.max
    if not os.path.isdir("./results/"):
        print("no results/ directory, starting from scratch")
        return
    for root, dirs, files in os.walk("./results/"):
        for filename in files:
            distance = float(os.path.splitext(filename)[0])
            if distance < bestDistance:
                best = filename
                bestDistance = distance
    if(best == ""):
        print("no solutions found from results/")
        return
    print("load solution from: results/"+best)
    file = open("./results/"+best, 'r')
    giftRun = GiftRun.Parse(file.readlines(), bestDistance)
    file.close()
    return giftRun


def updatePheromone(pheromone, giftRuns):
    pheromone = 0.9*pheromone
    for giftRun in giftRuns:
        for run in giftRun.all:
            lastIndex = KORVATUNTURI.index
            for nextPointId in run:
                nextIndex = int(nextPointId-1)
                pheromone[lastIndex, nextIndex] += 1 / (giftRun.totalDistance/9000000)
                lastIndex = nextIndex
            pheromone[lastIndex, KORVATUNTURI.index] += 1 / (giftRun.totalDistance / 9000000)
    return pheromone

def optimize(points, capacity, allDistances, iterations=1000):
    bestGiftRun = loadBest()
    if bestGiftRun is None:
        bestGiftRun = GiftRun()  # empty seed: deposits no pheromone
        bestGiftRun.totalDistance = math.inf
    iterGiftRuns = []
    attractiveness = 1 / allDistances
    pheromone = np.ones(allDistances.shape)

    n = 4
    iterGiftRuns = []
    for j in range(n):
        iterGiftRuns.append(bestGiftRun)

    start = timer()
    for i in range(iterations): # 30*4 = 120 ants, takes ~10mins
        pheromone = updatePheromone(pheromone, iterGiftRuns)
        print('pheromone updated - max: '+str(np.max(pheromone))+', min: '+str(np.min(pheromone)))
        #print(pheromone)
        if os.environ.get("SAVE_PHEROMONE"):  # 800 MB per iteration
            with open('pheromone.pickle', 'wb') as f:
                pickle.dump(pheromone, f)
            print('pheromone saved')
        queue = multiprocessing.Queue()
        jobs = []
        for j in range(n):
            ant = Ant(points, capacity, allDistances)
            process = multiprocessing.Process(target=ant.findSolution,
                                              args=(queue,attractiveness,pheromone))
            jobs.append(process)
        for job in jobs:
            job.start()
            print("Started")
        iterGiftRuns = []
        for j in range(n):
            newRun = queue.get()
            iterGiftRuns.append(newRun)
            current = timer()
            print("Run ("+str(i*4+j+1)+") has total distance: "+str(newRun.totalDistance)+" elapsed time: "+str(current-start))
            if newRun.totalDistance < bestGiftRun.totalDistance:
                print('New best found')
                bestGiftRun = newRun
                print('save solutions with total distance: ' + str(bestGiftRun.totalDistance))
                saveBest(bestGiftRun)
    return bestGiftRun


def solve(iterations=1000):
    points = readPoints()
    allDistances = calculateAllDistances(points)
    return optimize(points, 10000000, allDistances, iterations)