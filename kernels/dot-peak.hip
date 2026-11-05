// Стеля самих інструкцій добутку на gfx1030: fp16 / int8 / int4.
//
// Питання, яке це відокремлює: Triton-ядро int8 дало лише +7% над hipBLASLt
// fp16 при вдвічі вищій ПАСПОРТНІЙ стелі. Чи винна стеля (тобто драбина не
// працює на цьому чипі), чи ККД ядра? Тут немає ні пам'яті, ні тайлів —
// самі інструкції в регістрах, вісім незалежних накопичувачів, щоб залежності
// не гальмували конвеєр.
#include <hip/hip_runtime.h>
#include <cstdio>

#define ITERS 4096

template <int KIND>
__global__ void peak(const int* __restrict__ seed, int* __restrict__ out) {
  int a = seed[threadIdx.x & 63], b = seed[(threadIdx.x + 7) & 63];
  int A[8], B[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { A[i] = a + i; B[i] = b - i; }
  int acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float facc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (KIND == 0) {
        facc[i] = __builtin_amdgcn_fdot2(*(__attribute__((ext_vector_type(2))) _Float16*)&A[i],
                                         *(__attribute__((ext_vector_type(2))) _Float16*)&B[i],
                                         facc[i], false);
      } else if constexpr (KIND == 1) {
        acc[i] = __builtin_amdgcn_sdot4(A[i], B[i], acc[i], false);
      } else {
        acc[i] = __builtin_amdgcn_sdot8(A[i], B[i], acc[i], false);
      }
    }
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += (KIND == 0) ? (int)facc[i] : acc[i];
  if (threadIdx.x == 1023) out[blockIdx.x] = s;
}

int main() {
  hipDeviceProp_t p;
  hipGetDeviceProperties(&p, 0);
  int cu = p.multiProcessorCount;
  printf("пристрій: %s, CU: %d, такт: %d МГц\n", p.name, cu, p.clockRate / 1000);

  int *seed, *out;
  hipMalloc(&seed, 64 * sizeof(int));
  hipMalloc(&out, 4096 * sizeof(int));
  hipMemset(seed, 0x11, 64 * sizeof(int));

  const int block = 256, blocks = cu * 8;
  const char* nazvy[3] = {"fp16  (v_dot2_f32_f16)", "int8  (v_dot4_i32_i8)",
                          "int4  (v_dot8_i32_i4)"};
  const double mac[3] = {2.0, 4.0, 8.0};
  double baza = 0;
  for (int kind = 0; kind < 3; ++kind) {
    for (int warm = 0; warm < 3; ++warm) {
      if (kind == 0) peak<0><<<blocks, block>>>(seed, out);
      else if (kind == 1) peak<1><<<blocks, block>>>(seed, out);
      else peak<2><<<blocks, block>>>(seed, out);
    }
    hipDeviceSynchronize();
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    const int rep = 20;
    for (int r = 0; r < rep; ++r) {
      if (kind == 0) peak<0><<<blocks, block>>>(seed, out);
      else if (kind == 1) peak<1><<<blocks, block>>>(seed, out);
      else peak<2><<<blocks, block>>>(seed, out);
    }
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1);
    double sek = ms / 1e3 / rep;
    // множень = блоки × нитки × ітерації × 8 накопичувачів × MAC на інструкцію
    double mnozhen = (double)blocks * block * ITERS * 8 * mac[kind];
    double tops = mnozhen * 2.0 / sek / 1e12;  // множення+додавання
    if (kind == 0) baza = tops;
    printf("  %-24s %7.3f мс   %6.2f ТОпс   ×%.2f\n", nazvy[kind], ms / rep,
           tops, tops / baza);
  }
  return 0;
}
