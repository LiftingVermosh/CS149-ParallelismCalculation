#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#include "CycleTimer.h"

using namespace std;

enum ParallelMode {
  PARALLEL_ASSIGNMENTS = 0,
  PARALLEL_FULL = 1
};

typedef struct {
  // Control work assignments
  int start, end;     // 按 K 划分
  int startM, endM;   // 按 M 划分

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  double *partialCentroidSums;
  int *partialCentroidCounts;
  double *partialCost;
  int M, N, K;
  int threadId;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

// 优化 dist
double distSq(double *x, double *y, int nDim) {
    double accum = 0.0;
    for (int i = 0; i < nDim; i++) {
        double diff = x[i] - y[i];
        accum += diff * diff;
    }
    return accum;
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 * 
 *  核心问题：质心并行化导致同一时间的 minDist[m] 数据竞争
 * 
 *  考虑改为按 m 划分任务。每个线程负责一部分数据点
 */
void computeAssignments(WorkerArgs *const args) {
  double *minDist = new double[args->M];
  
  // Initialize arrays
  for (int m =0; m < args->M; m++) {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  // Assign datapoints to closest centroids
  for (int k = args->start; k < args->end; k++) {
    for (int m = 0; m < args->M; m++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist[m]) {
        minDist[m] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  delete[] minDist;
}

/* 并行执行的任务函数 */  
void computeAssignmentsParallel(WorkerArgs *const args) {
    for (int m = args->startM; m < args->endM; m++) {
        double minDist = 1e30;
        int closestCluster = -1;
        for (int k = 0; k < args->K; k++) {
            // 使用优化后的 distSq
            double d = distSq(&args->data[m * args->N],
                              &args->clusterCentroids[k * args->N], args->N);
            if (d < minDist) {
                minDist = d;
                closestCluster = k;
            }
        }
        args->clusterAssignments[m] = closestCluster;
    }
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }

  delete[] counts;
}

void computeCentroidPartials(WorkerArgs *const args) {
  double *localSums =
      &args->partialCentroidSums[args->threadId * args->K * args->N];
  int *localCounts = &args->partialCentroidCounts[args->threadId * args->K];

  for (int k = 0; k < args->K; k++) {
    localCounts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      localSums[k * args->N + n] = 0.0;
    }
  }

  for (int m = args->startM; m < args->endM; m++) {
    int k = args->clusterAssignments[m];
    localCounts[k]++;
    for (int n = 0; n < args->N; n++) {
      localSums[k * args->N + n] += args->data[m * args->N + n];
    }
  }
}

void computeCentroidsParallel(WorkerArgs *const args, int numThreads) {
  vector<double> partialSums(numThreads * args->K * args->N, 0.0);
  vector<int> partialCounts(numThreads * args->K, 0);
  vector<thread> threads(numThreads);
  vector<WorkerArgs> threadArgs(numThreads);

  int nPerThread = args->M / numThreads;
  for (int i = 0; i < numThreads; i++) {
    threadArgs[i] = *args;
    threadArgs[i].startM = i * nPerThread;
    threadArgs[i].endM = (i == numThreads - 1) ? args->M : (i + 1) * nPerThread;
    threadArgs[i].threadId = i;
    threadArgs[i].partialCentroidSums = partialSums.data();
    threadArgs[i].partialCentroidCounts = partialCounts.data();
    threads[i] = thread(computeCentroidPartials, &threadArgs[i]);
  }

  for (int i = 0; i < numThreads; i++) {
    threads[i].join();
  }

  for (int k = 0; k < args->K; k++) {
    int count = 0;
    for (int t = 0; t < numThreads; t++) {
      count += partialCounts[t * args->K + k];
    }

    count = max(count, 1);
    for (int n = 0; n < args->N; n++) {
      double sum = 0.0;
      for (int t = 0; t < numThreads; t++) {
        sum += partialSums[t * args->K * args->N + k * args->N + n];
      }
      args->clusterCentroids[k * args->N + n] = sum / count;
    }
  }
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  double *accum = new double[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += dist(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = 0; k < args->K; k++) {
    args->currCost[k] = accum[k];
  }

  delete[] accum;
}

void computeCostPartial(WorkerArgs *const args) {
  double *localCost = &args->partialCost[args->threadId * args->K];

  for (int k = 0; k < args->K; k++) {
    localCost[k] = 0.0;
  }

  for (int m = args->startM; m < args->endM; m++) {
    int k = args->clusterAssignments[m];
    localCost[k] += dist(&args->data[m * args->N],
                         &args->clusterCentroids[k * args->N], args->N);
  }
}

void computeCostParallel(WorkerArgs *const args, int numThreads) {
  vector<double> partialCost(numThreads * args->K, 0.0);
  vector<thread> threads(numThreads);
  vector<WorkerArgs> threadArgs(numThreads);

  int nPerThread = args->M / numThreads;
  for (int i = 0; i < numThreads; i++) {
    threadArgs[i] = *args;
    threadArgs[i].startM = i * nPerThread;
    threadArgs[i].endM = (i == numThreads - 1) ? args->M : (i + 1) * nPerThread;
    threadArgs[i].threadId = i;
    threadArgs[i].partialCost = partialCost.data();
    threads[i] = thread(computeCostPartial, &threadArgs[i]);
  }

  for (int i = 0; i < numThreads; i++) {
    threads[i].join();
  }

  for (int k = 0; k < args->K; k++) {
    double sum = 0.0;
    for (int t = 0; t < numThreads; t++) {
      sum += partialCost[t * args->K + k];
    }
    args->currCost[k] = sum;
  }
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon, int numThreads,
               int parallelMode) {

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args;
  args.data = data;
  args.clusterCentroids = clusterCentroids;
  args.clusterAssignments = clusterAssignments;
  args.currCost = currCost;
  args.partialCentroidSums = NULL;
  args.partialCentroidCounts = NULL;
  args.partialCost = NULL;
  args.M = M;
  args.N = N;
  args.K = K;
  args.threadId = 0;

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  // For time counting
  double totalAssignmentsTime = 0;
  double totalCentroidsTime = 0;
  double totalCostTime = 0;

  numThreads = max(1, min(numThreads, M));

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
        for (int k = 0; k < K; k++) prevCost[k] = currCost[k];
        double startAssignments = CycleTimer::currentSeconds();
        if (numThreads <= 1) {
            // Baseline - 原始串行
            args.start = 0; args.end = K;
            computeAssignments(&args); 
        } else {
            // 并行优化
            vector<thread> threads(numThreads);
            vector<WorkerArgs> threadArgs(numThreads);
            int nPerThread = M / numThreads;
            for (int i = 0; i < numThreads; i++) {
                threadArgs[i] = args;
                threadArgs[i].startM = i * nPerThread;
                threadArgs[i].endM = (i == numThreads - 1) ? M : (i + 1) * nPerThread;
                threads[i] = thread(computeAssignmentsParallel, &threadArgs[i]);
            }
            for (int i = 0; i < numThreads; i++) threads[i].join();
        }
        totalAssignmentsTime += (CycleTimer::currentSeconds() - startAssignments);

        double startCentroids = CycleTimer::currentSeconds();
        if (parallelMode == PARALLEL_FULL && numThreads > 1) {
          computeCentroidsParallel(&args, numThreads);
        } else {
          computeCentroids(&args);
        }
        totalCentroidsTime += (CycleTimer::currentSeconds() - startCentroids);

        double startCost = CycleTimer::currentSeconds();
        if (parallelMode == PARALLEL_FULL && numThreads > 1) {
          computeCostParallel(&args, numThreads);
        } else {
          computeCost(&args);
        }
        totalCostTime += (CycleTimer::currentSeconds() - startCost);

        iter++;
  }

  printf("Performance Summary (Total %d iterations):\n", iter);
  printf("  Assignments: %.4f s\n", totalAssignmentsTime);
  printf("  Centroids:   %.4f s\n", totalCentroidsTime);
  printf("  Cost:        %.4f s\n", totalCostTime);
  delete[] currCost;
  delete[] prevCost;
}
