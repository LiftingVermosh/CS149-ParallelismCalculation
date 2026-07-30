#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

/* Include in Version 3 for exclusive scan */
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];

/* Self-use MARCO */
#define TILE_NUM 16
#define SHARED_MEM_BATCH 128 

#ifndef RENDER_VERSION
#define RENDER_VERSION 4
#endif

// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

/* Help Functions */

// 返回是否在圆内
__device__ bool pixelInCircle(float dist, float rad) {
    return dist <= rad;
}


// 计算指定点间的距离
__device__ __inline__ float sqDist(float2 p1, float3 p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

/* Task 3 realizes Version 1 */ 

// kernelRenderPixels -- (CUDA device code)
//
// Each thread renders a pixel. Note that written by pixel avoiding write conflict
__global__ void kernelRenderPixels() {
    int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelY = blockDim.y * blockIdx.y + threadIdx.y;

    int imageHeight = cuConstRendererParams.imageHeight;
    int imageWidth = cuConstRendererParams.imageWidth;
    float invHeight = 1.f / imageHeight;
    float invWidth = 1.f / imageWidth;

    // 边界检查
    if(pixelX >= imageWidth || pixelY >= imageHeight) return;

    float4 pixelColor = make_float4(0.f, 0.f, 0.f, 1.f);
    float *imageData = cuConstRendererParams.imageData;

    float2 pixelCenterNorm = make_float2(
        invWidth * (static_cast<float>(pixelX) + 0.5f),
        invHeight * (static_cast<float>(pixelY) + 0.5f)
    );

    // 读取背景颜色
    int imgIdx = 4 * (pixelY * imageWidth + pixelX);
    float4* imgPtr = (float4*)(&imageData[imgIdx]);
    float4 localColor = *imgPtr; 

    // 遍历所有圆
    int numCircles = cuConstRendererParams.numCircles;
    for(int i = 0; i < numCircles; ++i) {
        float3 p = *(float3*)(&cuConstRendererParams.position[i * 3]); 
        float rad = cuConstRendererParams.radius[i];

        float distL2 = sqDist(pixelCenterNorm, p);

        if(pixelInCircle(distL2, rad * rad)) {
            shadePixel(i, pixelCenterNorm, p, &localColor);
        }
    }

    *imgPtr = localColor;
}

/* Task 3 realizes Version 2 */

// kernelRenderPixelsWithMapping -- (CUDA device code)
//
// Each thread renders a pixel. Note that written by pixel avoiding write conflict
// Comapred with Version 1, we use preprocessing mapping (on cpu) here INSTEAD OF fully scan circles strategy
__global__ void kernelRenderPixelsWithMapping(
        int* tileCounts, 
        int* tileOffsets, 
        int* circleIndices
    ) {
    int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelY = blockDim.y * blockIdx.y + threadIdx.y;

    int tileIdx = blockIdx.y * gridDim.x + blockIdx.x;
    int startOffset = tileOffsets[tileIdx];
    int count = tileCounts[tileIdx];

    int imageHeight = cuConstRendererParams.imageHeight;
    int imageWidth = cuConstRendererParams.imageWidth;
    float invHeight = 1.f / imageHeight;
    float invWidth = 1.f / imageWidth;

    // 边界检查
    if(pixelX >= imageWidth || pixelY >= imageHeight) return;

    float4 pixelColor = make_float4(0.f, 0.f, 0.f, 1.f);
    float *imageData = cuConstRendererParams.imageData;

    float2 pixelCenterNorm = make_float2(
        invWidth * (static_cast<float>(pixelX) + 0.5f),
        invHeight * (static_cast<float>(pixelY) + 0.5f)
    );

    // 读取背景颜色
    int imgIdx = 4 * (pixelY * imageWidth + pixelX);
    float4* imgPtr = (float4*)(&imageData[imgIdx]);
    float4 localColor = *imgPtr; 

    // 改进点：采用预处理映射表
    for (int i = 0; i < count; i++) {
        int circleIdx = circleIndices[startOffset + i];
        
        float3 p = *(float3*)(&cuConstRendererParams.position[circleIdx * 3]);
        float rad = cuConstRendererParams.radius[circleIdx];
        
        float distL2 = sqDist(pixelCenterNorm, p);

        // 分块在不等于像素在，所以还要 check
        if (pixelInCircle(distL2, rad * rad)) {
            shadePixel(circleIdx, pixelCenterNorm, p, &localColor);
        }
    }

    *imgPtr = localColor;
}

/* Task 3 realizes Version 3 */

// kernelCountCirclesPerTile -- (CUDA device code)
//
// 以圆为线程处理单位计算每个 Tile 包含的圆数
__global__ void kernelCountCirclesPerTile(int* deviceTileCounts, int gridX, int gridY) {
    // 当前线程处理圆的 id
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    // 超过圆的个数
    if(index >= cuConstRendererParams.numCircles) {
        return;
    }
    
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    float3 p = *(float3*)(&cuConstRendererParams.position[3 * index]);
    float rad = cuConstRendererParams.radius[index];
    
    // 计算圆在 Tile 坐标系下的范围
    int minX = max(0        , static_cast<int>((p.x - rad) * imageWidth) / TILE_NUM);
    int maxX = min(gridX - 1, static_cast<int>((p.x + rad) * imageWidth) / TILE_NUM);
    int minY = max(0        , static_cast<int>((p.y - rad) * imageHeight) / TILE_NUM);
    int maxY = min(gridY - 1, static_cast<int>((p.y + rad) * imageHeight) / TILE_NUM);
    
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            int tileIdx = y * gridX + x;
            atomicAdd(&deviceTileCounts[tileIdx], 1);  // 使用原子加法避免数据竞争
        }
    }
}

// kernelFillCircleIndices -- (CUDA device code)
//
// 以圆为线程处理单位写回 Flatten Array
__global__ void kernelFillCircleIndices(
        int* deviceTileOffsets,  int* deviceCircleIndices, 
        int* deviceTempCounters, 
        int gridX, int gridY
    ){
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= cuConstRendererParams.numCircles) {
            return;
        }
        
        int imageWidth = cuConstRendererParams.imageWidth;
        int imageHeight = cuConstRendererParams.imageHeight;
        
        float3 p = *(float3*)(&cuConstRendererParams.position[3 * index]);
        float rad = cuConstRendererParams.radius[index];
        
        int minX = max(0, (int)((p.x - rad) * imageWidth) / TILE_NUM);
        int maxX = min(gridX - 1, (int)((p.x + rad) * imageWidth) / TILE_NUM);
        int minY = max(0, (int)((p.y - rad) * imageHeight) / TILE_NUM);
        int maxY = min(gridY - 1, (int)((p.y + rad) * imageHeight) / TILE_NUM);
        
        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                int tileIdx = y * gridX + x;
                // 原子操作获取当前圆在该 Tile 列表中的相对偏移
                int entryIdx = atomicAdd(&deviceTempCounters[tileIdx], 1);
                deviceCircleIndices[deviceTileOffsets[tileIdx] + entryIdx] = index;  // 写入绝对位置
            }
        }
}

// kernelSortTileIndices -- (CUDA device code)
//
// Alpha Blending 不满足交换律，因此需要确保每个 Tile 内部的圆索引是升序排列的
__global__ void kernelSortTileIndices(int* tileCounts, int* tileOffsets, int* circleIndices, int numTiles) {
    int tileIdx = blockIdx.y * gridDim.x + blockIdx.x;
    if (tileIdx >= numTiles) return;

    // 只允许 Block 内的第一个线程进行排序,否则 Race Condition 白排序
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int start = tileOffsets[tileIdx];
        int count = tileCounts[tileIdx];
        
        for (int i = 1; i < count; i++) {
            int key = circleIndices[start + i];
            int j = i - 1;
            while (j >= 0 && circleIndices[start + j] > key) {
                circleIndices[start + j + 1] = circleIndices[start + j];
                j--;
            }
            circleIndices[start + j + 1] = key;
        }
    }
}

/* Task 4 realize Version 4 */
__global__ void kernelRenderPixelsWithSharedMem(
    int* tileCounts, 
    int* tileOffsets, 
    int* circleIndices
) {
    // 声明共享内存
    __shared__ float3 sPos[SHARED_MEM_BATCH];
    __shared__ float  sRad[SHARED_MEM_BATCH];
    __shared__ int    sCircleIdx[SHARED_MEM_BATCH];

    int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelY = blockDim.y * blockIdx.y + threadIdx.y;
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x; // Block 内线性索引
    int tileIdx = blockIdx.y * gridDim.x + blockIdx.x;
    int totalInTile = tileCounts[tileIdx];
    int startOffset = tileOffsets[tileIdx];

    // 基础参数计算
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    float2 pixelCenterNorm = make_float2(
        invWidth * (static_cast<float>(pixelX) + 0.5f),
        invHeight * (static_cast<float>(pixelY) + 0.5f)
    );

    // 读取背景颜色
    float4 localColor = make_float4(0.f, 0.f, 0.f, 0.f); 
    bool outOfBounds = (pixelX >= imageWidth || pixelY >= imageHeight);
    if (!outOfBounds) {
        localColor = *(float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    }

    for (int batchStart = 0; batchStart < totalInTile; batchStart += SHARED_MEM_BATCH) {
        
        int curBatchSize = min(SHARED_MEM_BATCH, totalInTile - batchStart);
        // 协作搬运圆的数据
        for (int i = localIdx; i < curBatchSize; i += blockDim.x * blockDim.y) {
            int cIdx = circleIndices[startOffset + batchStart + i];
            sCircleIdx[i] = cIdx;
            sPos[i] = *(float3*)(&cuConstRendererParams.position[cIdx * 3]);
            sRad[i] = cuConstRendererParams.radius[cIdx];
        }
        // 同步确保所有线程都搬完了
        __syncthreads();
        // 像素判定与上色
        if (!outOfBounds) {
            for (int i = 0; i < curBatchSize; i++) {
                float3 p = sPos[i];
                float rad = sRad[i];
                float distL2 = sqDist(pixelCenterNorm, p);
                if (pixelInCircle(distL2, rad * rad)) {
                    shadePixel(sCircleIdx[i], pixelCenterNorm, p, &localColor);
                }
            }
        }
        // 进入下一批前再次同步，防止上一批数据被覆盖
        __syncthreads();
    }
    // 写回 Global Memory
    if (!outOfBounds) {
        int imgIdx = 4 * (pixelY * imageWidth + pixelX);
        *(float4*)(&cuConstRendererParams.imageData[imgIdx]) = localColor;
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;

    /* Updated 4 Version 2 */
    hostCircleIndices = NULL;   
    hostTileCounts = NULL;     
    hostTileOffsets = NULL;    
    deviceCircleIndices = NULL;
    deviceTileCounts = NULL;    
    deviceTileOffsets = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }

    /* Update 4 Version 2 */
    if (hostCircleIndices) {
        delete [] hostCircleIndices;
        delete [] hostTileCounts;
        delete [] hostTileOffsets;
    }

    if(deviceCircleIndices) {
        cudaFree(deviceCircleIndices);
        cudaFree(deviceTileCounts);
        cudaFree(deviceTileOffsets);
        cudaFree(deviceTempCounters);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene, int seed) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius, seed);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

    /* Update 4 Version 2 */
    int gridX = (image->width + TILE_NUM - 1) / TILE_NUM;
    int gridY = (image->height + TILE_NUM - 1) / TILE_NUM;
    int numTiles = gridX * gridY;

    // 分配 Tile 统计信息空间
    hostTileCounts = new int[numTiles];
    hostTileOffsets = new int[numTiles];
    cudaMalloc(&deviceTileCounts, sizeof(int) * numTiles);
    cudaMalloc(&deviceTileOffsets, sizeof(int) * numTiles);
    
    // 因为 `CircleIndices` 的大小取决于圆的重叠情况，所以初始化 `deviceCircleIndices` 为 NULL
    // 交由 `render()` 动态分配
    deviceCircleIndices = NULL;
    
    /* Update 4 Version 3 */
    // Compromise of dyamnic allocation
    indicesCapacity = numCircles * 32; 
    cudaMalloc(&deviceCircleIndices, sizeof(int) * indicesCapacity);
    cudaMalloc(&deviceTempCounters, sizeof(int) * numTiles);
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void CudaRenderer::buildTileCircleMapping() {
    int gridX = (this->image->width + TILE_NUM - 1) / TILE_NUM;
    int gridY = (this->image->height + TILE_NUM - 1) / TILE_NUM;

    int numTiles = gridX * gridY;

    memset(hostTileCounts, 0, sizeof(int) * numTiles);

    struct Box { int minX, minY, maxX, maxY; };
    std::vector<Box> circleBoxes(this->numCircles);


    for (int i = 0; i < numCircles; i++) {
        float3 p = *(float3*)(&this->position[3 * i]);
        float rad = radius[i];
        
        // 计算圆在 Tile 坐标系下的范围
        circleBoxes[i].minX = std::max(0        , static_cast<int>((p.x - rad) * this->image->width) / TILE_NUM);
        circleBoxes[i].maxX = std::min(gridX - 1, static_cast<int>((p.x + rad) * this->image->width) / TILE_NUM);
        circleBoxes[i].minY = std::max(0        , static_cast<int>((p.y - rad) * this->image->height) / TILE_NUM);
        circleBoxes[i].maxY = std::min(gridY - 1, static_cast<int>((p.y + rad) * this->image->height) / TILE_NUM);
        
        for (int y = circleBoxes[i].minY; y <= circleBoxes[i].maxY; y++) {
            for (int x = circleBoxes[i].minX; x <= circleBoxes[i].maxX; x++) {
                hostTileCounts[y * gridX + x]++;
            }
        }
    }

    // 计算偏移量
    int totalIndices = 0;
    for (int i = 0; i < numTiles; i++) {
        hostTileOffsets[i] = totalIndices;
        totalIndices += hostTileCounts[i];
    }

    // 管理 CircleIndices 显存空间
    static int currentIndicesCapacity = 0;
    if (totalIndices > currentIndicesCapacity) {
        
        if (deviceCircleIndices) {
            cudaFree(deviceCircleIndices);
        }

        if (hostCircleIndices) {
            delete[] hostCircleIndices;
        }

        cudaMalloc(&deviceCircleIndices, sizeof(int) * totalIndices);
        hostCircleIndices = new int[totalIndices];

        currentIndicesCapacity = totalIndices;
    }

    // 第二趟扫描：填充索引
    std::vector<int> tempOffsets(numTiles, 0);  // 一个临时的偏移数组来记录当前填到哪里了
    for (int i = 0; i < numCircles; i++) {
        for (int y = circleBoxes[i].minY; y <= circleBoxes[i].maxY; y++) {
            for (int x = circleBoxes[i].minX; x <= circleBoxes[i].maxX; x++) {
                int tileIdx = y * gridX + x;
                int writePos = hostTileOffsets[tileIdx] + tempOffsets[tileIdx];
                hostCircleIndices[writePos] = i;
                tempOffsets[tileIdx]++;
            }
        }
    }
    
    // 拷贝到 GPU
    cudaMemcpy(deviceTileCounts, hostTileCounts, sizeof(int) * numTiles, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceTileOffsets, hostTileOffsets, sizeof(int) * numTiles, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCircleIndices, hostCircleIndices, sizeof(int) * totalIndices, cudaMemcpyHostToDevice);
}

void
CudaRenderer::render() {

#if RENDER_VERSION == 1
    /* Improved Version #1 - Pixel Unit Based */
    int imageWidth = image->width;
    int imageHeight = image->height;

    dim3 blockDim(TILE_NUM, TILE_NUM);
    dim3 gridDim(
        (imageWidth + blockDim.x - 1) / blockDim.x,
        (imageHeight + blockDim.y - 1) / blockDim.y
    );

    kernelRenderPixels<<<gridDim, blockDim>>>();

#elif RENDER_VERSION == 2
    /* Improved Version #2 - Pixel Unit Based & Traverse Optimized */
    buildTileCircleMapping();

    int imageWidth = image->width;
    int imageHeight = image->height;

    dim3 blockDim(TILE_NUM, TILE_NUM);
    dim3 gridDim(
        (imageWidth + blockDim.x - 1) / blockDim.x,
        (imageHeight + blockDim.y - 1) / blockDim.y
    );

    kernelRenderPixelsWithMapping<<<gridDim, blockDim>>>(
        deviceTileCounts,
        deviceTileOffsets,
        deviceCircleIndices
    );

#elif RENDER_VERSION == 3 || RENDER_VERSION == 4
    /* Improved Version #3/#4 - Pure GPU Preprocess */

    int gridX = (image->width + TILE_NUM - 1) / TILE_NUM;
    int gridY = (image->height + TILE_NUM - 1) / TILE_NUM;
    int numTiles = gridX * gridY;

    // 清零计数器
    cudaMemset(deviceTileCounts, 0, sizeof(int) * numTiles);

    // 在圆视角下建立映射
    dim3 blockDimCircle(256);
    dim3 gridDimCircle((numCircles + blockDimCircle.x - 1) / blockDimCircle.x);
    kernelCountCirclesPerTile<<<gridDimCircle, blockDimCircle>>>(deviceTileCounts, gridX, gridY);

    // 前缀和计算
    // v1 - 库函数
    thrust::device_ptr<int> d_counts(deviceTileCounts);
    thrust::device_ptr<int> d_offsets(deviceTileOffsets);

    thrust::exclusive_scan(d_counts, d_counts + numTiles, d_offsets);
    // TODO : v2 - 自实现前缀和

    // 可容纳性检查
    int totalIndices;
    int lastOffset, lastCount;
    // 获取最后一个 offset + 最后一个 count
    cudaMemcpy(&lastOffset, deviceTileOffsets + numTiles - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastCount, deviceTileCounts + numTiles - 1, sizeof(int), cudaMemcpyDeviceToHost);
    totalIndices = lastOffset + lastCount;
    // 如果当前容量不够，再重新分配
    if (totalIndices > indicesCapacity) {
        cudaFree(deviceCircleIndices);
        indicesCapacity = 1.2f * totalIndices;
        cudaMalloc(&deviceCircleIndices, sizeof(int) * indicesCapacity);
    }

    // 写回索引
    cudaMemset(deviceTempCounters, 0, sizeof(int) * numTiles);
    kernelFillCircleIndices<<<gridDimCircle, blockDimCircle>>>(
        deviceTileOffsets, deviceCircleIndices, 
        deviceTempCounters, 
        gridX, gridY
    );

    dim3 blockDimRender(TILE_NUM, TILE_NUM);
    dim3 gridDimRender(gridX, gridY);

    // 排序避免顺序错位
    kernelSortTileIndices<<<gridDimRender, blockDimRender>>>(
        deviceTileCounts, 
        deviceTileOffsets, 
        deviceCircleIndices, 
        numTiles
    );

#if RENDER_VERSION == 3
    kernelRenderPixelsWithMapping<<<gridDimRender, blockDimRender>>>(
        deviceTileCounts,
        deviceTileOffsets,
        deviceCircleIndices
    );
#else
    kernelRenderPixelsWithSharedMem<<<gridDimRender, blockDimRender>>>(
        deviceTileCounts, 
        deviceTileOffsets, 
        deviceCircleIndices
    );
#endif

#else
#error "Unsupported RENDER_VERSION. Use 1, 2, 3, or 4."
#endif

    cudaDeviceSynchronize();
}
