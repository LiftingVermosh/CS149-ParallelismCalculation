#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#include "circleRenderer.h"

class CudaRenderer : public CircleRenderer {
 private:
  Image* image;
  SceneName sceneName;

  int numCircles;
  float* position;
  float* velocity;
  float* color;
  float* radius;

  float* cudaDevicePosition;
  float* cudaDeviceVelocity;
  float* cudaDeviceColor;
  float* cudaDeviceRadius;
  float* cudaDeviceImageData;

  /* Updated 4 Version 2 */
  int* hostCircleIndices;   // 存储圆索引的 Flatten 数组
  int* hostTileCounts;      // 每个 Tile 包含的圆数量
  int* hostTileOffsets;     // 每个 Tile 在索引数组中的起始偏移
  // 对应的 GPU 显存
  int* deviceCircleIndices;
  int* deviceTileCounts;    
  int* deviceTileOffsets;

  /* Updated 4 Version 3 */
  int indicesCapacity;      // 空间预分配
  int* deviceTempCounters;  // 临时计数器

 public:
  CudaRenderer();
  virtual ~CudaRenderer();

  const Image* getImage();

  void setup();

  void loadScene(SceneName name, int seed = 0);

  void allocOutputImage(int width, int height);

  void clearImage();

  void advanceAnimation();

  void render();

  void shadePixel(int circleIndex, float pixelCenterX, float pixelCenterY,
                  float px, float py, float pz, float* pixelData);

  /* Updated 4 Version 2 */
  void buildTileCircleMapping();
};

#endif