#include <algorithm>
#include <iostream>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>

#include "CycleTimer.h"

#define SEED 7
#define SAMPLE_RATE 1e-2

using namespace std;

// Main compute functions
extern void kMeansThread(double *data, double *clusterCentroids,
                      int *clusterAssignments, int M, int N, int K,
                      double epsilon, int numThreads, int parallelMode);
extern double dist(double *x, double *y, int nDim);

// Utilities
extern void logToFile(string filename, double sampleRate, double *data,
                      int *clusterAssignments, double *clusterCentroids, int M,
                      int N, int K);
extern void writeData(string filename, double *data, double *clusterCentroids,
                      int *clusterAssignments, int *M_p, int *N_p, int *K_p,
                      double *epsilon_p);
extern void readData(string filename, double **data, double **clusterCentroids,
                     int **clusterAssignments, int *M_p, int *N_p, int *K_p,
                     double *epsilon_p);

// Functions for generating data
double randDouble() {
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

void initData(double *data, int M, int N) {
  int K = 10;
  double *centers = new double[K * N];

  // Gaussian noise
  double mean = 0.0;
  double stddev = 0.5;
  std::default_random_engine generator;
  std::normal_distribution<double> normal_dist(mean, stddev);

  // Randomly create points to center data around
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      centers[k * N + n] = randDouble();
    }
  }

  // Even clustering
  for (int m = 0; m < M; m++) {
    int startingPoint = rand() % K; // Which center to start from
    for (int n = 0; n < N; n++) {
      double noise = normal_dist(generator);
      data[m * N + n] = centers[startingPoint * N + n] + noise;
    }
  }

  delete[] centers;
}

void initCentroids(double *clusterCentroids, int K, int N) {
  // Initialize centroids (close together - makes it a bit more interesting)
  for (int n = 0; n < N; n++) {
    clusterCentroids[n] = randDouble();
  }
  for (int k = 1; k < K; k++) {
    for (int n = 0; n < N; n++) {
      clusterCentroids[k * N + n] =
          clusterCentroids[n] + (randDouble() - 0.5) * 0.1;
    }
  }
}

int main(int argc, char** argv) {
  srand(SEED);

  int M, N, K;
  double epsilon;

  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  int numThreads = 1;
  int parallelMode = 0; // 0: assignments only, 1: full
  string modeName = "assignments";
  int opt;
  while ((opt = getopt(argc, argv, "t:m:")) != -1) {
    switch (opt) {
      case 't':
        numThreads = atoi(optarg);
        break;
      case 'm':
        modeName = string(optarg);
        if (modeName == "assignments" || modeName == "assignment" ||
            modeName == "assign") {
          parallelMode = 0;
          modeName = "assignments";
        } else if (modeName == "full") {
          parallelMode = 1;
        } else {
          fprintf(stderr, "Unknown mode: %s\n", optarg);
          fprintf(stderr,
                  "Usage: %s [-t numThreads] [-m assignments|full]\n",
                  argv[0]);
          exit(EXIT_FAILURE);
        }
        break;
      default:
        fprintf(stderr, "Usage: %s [-t numThreads] [-m assignments|full]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  // NOTE: we will grade your submission using the data in data.dat
  // which is read by this function
  readData("./data/data.dat", &data, &clusterCentroids, &clusterAssignments, &M, &N,
           &K, &epsilon);

  // NOTE: if you want to generate your own data (for fun), you can use the
  // below code
  /*
  M = 1e6;
  N = 100;
  K = 3;
  epsilon = 0.1;

  data = new double[M * N];
  clusterCentroids = new double[K * N];
  clusterAssignments = new int[M];

  // Initialize data
  initData(data, M, N);
  initCentroids(clusterCentroids, K, N);

  // Initialize cluster assignments
  for (int m = 0; m < M; m++) {
    double minDist = 1e30;
    int bestAssignment = -1;
    for (int k = 0; k < K; k++) {
      double d = dist(&data[m * N], &clusterCentroids[k * N], N);
      if (d < minDist) {
        minDist = d;
        bestAssignment = k;
      }
    }
    clusterAssignments[m] = bestAssignment;
  }

  // Uncomment to generate data file
  // writeData("./data.dat", data, clusterCentroids, clusterAssignments, &M, &N,
  //           &K, &epsilon);
  */

  printf("Running K-means with: M=%d, N=%d, K=%d, epsilon=%f, Threads=%d, Mode=%s\n",
         M, N, K, epsilon, numThreads, modeName.c_str());

  // Log the starting state of the algorithm
  logToFile("./logs/start.log", SAMPLE_RATE, data, clusterAssignments,
            clusterCentroids, M, N, K);

  double startTime = CycleTimer::currentSeconds();
  kMeansThread(data, clusterCentroids, clusterAssignments, M, N, K, epsilon,
               numThreads, parallelMode);
  double endTime = CycleTimer::currentSeconds();
  printf("[Total Time]: %.3f ms\n", (endTime - startTime) * 1000);

  // Log the end state of the algorithm
  logToFile("./logs/end.log", SAMPLE_RATE, data, clusterAssignments,
            clusterCentroids, M, N, K);

  delete[] data;
  delete[] clusterCentroids;
  delete[] clusterAssignments;
  return 0;
}
