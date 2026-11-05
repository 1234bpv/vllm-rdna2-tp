// Квантування активацій у 4 біти для ядра W4A4: fp16 -> беззнакові напівбайти
// з перемішуванням exllama, плюс суми по групах.
//
// На кожен рядок один масштаб: sa = max|x| / 7 (значення a лежать у -8..7,
// беремо 7 задля запасу). Пакуємо a' = round(x/sa) + 8 у 0..15.
// Порядок напівбайтів — той самий exllama [0,2,4,6,1,3,5,7], бо ваги лежать
// так, а добуток пов'язує однакові позиції.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define GS 128
#define WORDS (GS / 8)

// Один блок на рядок; 256 ниток проходять рядок смугами.
__global__ __launch_bounds__(256) void quant_rows(
    const __half* __restrict__ X, int* __restrict__ Aq,
    float* __restrict__ SAp, float* __restrict__ Sa, int M, int K) {
  const int m = blockIdx.x;
  const int K8 = K / 8, NG = K / GS;
  __shared__ float red[256];

  // 1. Максимум за модулем по рядку.
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += 256)
    mx = fmaxf(mx, fabsf(__half2float(X[(long)m * K + k])));
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float sa = fmaxf(red[0], 1e-8f) / 7.0f;
  const float inv = 1.0f / sa;
  if (threadIdx.x == 0) Sa[m] = sa;

  // 2. Пакування й суми по групах.
  //    Порядок напівбайтів exllama: позиція p слова несе k = shuf[p].
  const int shuf[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  for (int w = threadIdx.x; w < K8; w += 256) {
    int word = 0, s = 0;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      int k = w * 8 + shuf[p];
      float v = __half2float(X[(long)m * K + k]) * inv;
      int q = (int)rintf(v) + 8;
      q = q < 0 ? 0 : (q > 15 ? 15 : q);
      word |= q << (4 * p);
      s += q;
    }
    Aq[(long)m * K8 + w] = word;
    atomicAdd(&SAp[(long)m * NG + (w * 8) / GS], (float)s);
  }
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int K = argc > 2 ? atoi(argv[2]) : 5120;
  int K8 = K / 8, NG = K / GS;
  std::vector<__half> hX((long)M * K);
  for (auto& v : hX) v = __float2half((rand() % 2000 - 1000) * 0.001f);
  __half* dX; int* dAq; float *dSAp, *dSa;
  hipMalloc(&dX, hX.size() * 2); hipMalloc(&dAq, (long)M * K8 * 4);
  hipMalloc(&dSAp, (long)M * NG * 4); hipMalloc(&dSa, M * 4);
  hipMemcpy(dX, hX.data(), hX.size() * 2, hipMemcpyHostToDevice);
  for (int i = 0; i < 5; ++i) {
    hipMemset(dSAp, 0, (long)M * NG * 4);
    quant_rows<<<M, 256>>>(dX, dAq, dSAp, dSa, M, K);
  }
  hipDeviceSynchronize();
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0);
  for (int r = 0; r < 20; ++r) {
    hipMemset(dSAp, 0, (long)M * NG * 4);
    quant_rows<<<M, 256>>>(dX, dAq, dSAp, dSa, M, K);
  }
  hipEventRecord(e1); hipEventSynchronize(e1);
  float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 20;
  printf("квантування активацій M=%d K=%d: %.4f мс\n", M, K, ms);
  printf("  для порівняння, саме множення при N=17408: 7,47 мс -> надбавка %.1f%%\n",
         100.0 * ms / 7.47);
  return 0;
}
