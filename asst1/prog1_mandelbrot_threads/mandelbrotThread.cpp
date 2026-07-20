#include <stdio.h>
#include <thread>
#include <cstdlib>  

#include "CycleTimer.h"

enum MODE{
    SERIAL = 1,
    INTERLEAVED = 2
};

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
    int startRows;
    int numRows;
    int strides;
    int mode;
} WorkerArgs;


extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

extern void mandelbrotInterleaved(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int stride,
    int maxIterations,
    int output[]);

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs * const args) {

    // TODO FOR CS149 STUDENTS: Implement the body of the worker
    // thread here. Each thread should make a call to mandelbrotSerial()
    // to compute a part of the output image.  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.

    // printf("Hello world from thread %d\n", args->threadId);
    if (args->mode == SERIAL) {
        mandelbrotSerial(
            args->x0, args->y0, args->x1, args->y1,
            args->width, args->height,
            args->startRows, args->numRows,
            args->maxIterations, args->output
        );
    } else if (args->mode == INTERLEAVED) {
        mandelbrotInterleaved(
            args->x0, args->y0, args->x1, args->y1,
            args->width, args->height,
            args->startRows, args->strides,
            args->maxIterations, args->output
        );
    }
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int mode, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
      
        // TODO FOR CS149 STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;
      
        args[i].threadId = i;
        args[i].mode = mode;

        // Calculate needed rows
        if (args[i].mode == SERIAL) {               // Version 1 - Direct Split
            args[i].numRows = args[i].height / args[i].numThreads;
            args[i].startRows = args[i].numRows * args[i].threadId;
            if(args[i].numThreads == args[i].threadId + 1) {
                args[i].numRows = args[i].height - args[i].startRows;
            }
        } else if (args[i].mode == INTERLEAVED) {   // Version 2 - Cross Allocation
            args[i].startRows = i; 
            args[i].strides = numThreads;
        }
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i=1; i<numThreads; i++) {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }
    
    workerThreadStart(&args[0]);

    // join worker threads
    for (int i=1; i<numThreads; i++) {
        workers[i].join();
    }
}

