// Ядро W4A8 для AMD gfx1030 (RDNA2): ваги 4 біти (беззнакові, 0..15 з нульовими точками),
// активації 8 бітів (ЗНАКОВІ, -127..127), групове квантування (G = 128).
//
// ── Математика квантування та згортки ──
// Активація: x_k = a_k * sa_g, де a_k ∈ [-127, 127] (signed 8-bit),
//             sa_g — масштаб активацій на групу g (sa_g = max|x| / 127).
// Вага:      W_k = (q_k - zp_g) * sw_g, де q_k ∈ [0, 15] (unsigned 4-bit),
//             zp_g — нульова точка ваг на групу g (0..15),
//             sw_g — масштаб ваг на групу g.
// G = 128 — розмір групи вздовж K.
//
// Зсуваємо ваги та нульові точки у знаковий діапазон [-8, 7]:
//   q_s,k = q_k - 8 ∈ [-8, 7],   zp_s,g = zp_g - 8 ∈ [-8, 7]
// Тоді:
//   W_k = (q_s,k - zp_s,g) * sw_g
//
// Для однієї групи g (індекси k від 0 до G - 1):
//   Σ_{k=0}^{G-1} x_k * W_k = Σ a_k * sa_g * (q_s,k - zp_s,g) * sw_g
//     = sa_g * sw_g * Σ [ a_k * q_s,k - a_k * zp_s,g ]
//     = sa_g * sw_g * [ Σ (a_k * q_s,k) - zp_s,g * (Σ a_k) ]
//     = sa_g * sw_g * [ sdot4(a, q_s) - zp_s,g * Sa_g ]
//
// де:
//   Sa_g = Σ a_k — знакова сума активацій по групі (залежить від рядка m і групи g),
//   sdot4(a, q_s) = Σ_{k=0}^{G-1} a_k * q_s,k обчислюється через апаратну інструкцію v_dot4_i32_i8:
//   __builtin_amdgcn_sdot4(a, b, acc, false).
//
// Поправка E на боці ваг ПОВНІСТЮ ВІДСУТНЯ, що економить пам'ять та LDS.
//
// Накопичення у вихідний акумулятор:
//   v = (float)acc - zp_s,g * Sa_g;
//   fout += (sw_g * sa_g) * v;
//
// ── Розкладка в LDS та розпакування ──
// 1. Активації в LDS: група зі 128 значень = 32 слова int32 (4 знакові байти на слово).
// 2. Ваги в глобальній пам'яті: 16 слів int32 (8 півбайтів на слово). При записі в LDS
//    ваги розпаковуються у знакові байти через віднімання 8:
//    байт_j = ((півбайт_j - 8) & 0xFF).
//    Група розпакованих ваг у LDS займає 32 слова int32.
// 3. Загальний бюджет LDS: 17408 + 17408 + 2048 = 36 864 Б (36 КБ).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define GS 128
#define WORDS (GS / 4)        // 32 слова int32 у LDS на групу (128 байтів / 4)
#define WORDS_RAW (GS / 8)    // 16 слів int32 у пам'яті для 4-бітних ваг

typedef int int2v __attribute__((ext_vector_type(2)));
typedef int int4v __attribute__((ext_vector_type(4)));

// Розпакування 8 півбайтів (4-бітних значень 0..15) із 32-бітного слова v у два 32-бітні слова
// по 4 знакові байти (-8..7) у доповняльному коді: байт_j = ((півбайт_j − 8) & 0xFF).
// Віднімання 8 виконується від кожного півбайта окремо перед зсувом:
// (байт_j = ((півбайт_j − 8) & 0xFF)).
// Зауважимо: скалярне віднімання 32-бітного слова (lo - 0x08080808) не застосовується,
// оскільки для півбайтів < 8 у 32-бітній арифметиці виникає міжбайтова позика (borrow) у старші байти.
static inline __device__ void unpack_nibbles(int v, int& lo, int& hi) {
  int b0 = (v & 0xF) - 8;
  int b1 = ((v >> 4) & 0xF) - 8;
  int b2 = ((v >> 8) & 0xF) - 8;
  int b3 = ((v >> 12) & 0xF) - 8;
  lo = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);

  unsigned int vh = (unsigned int)v >> 16;
  int b4 = (vh & 0xF) - 8;
  int b5 = ((vh >> 4) & 0xF) - 8;
  int b6 = ((vh >> 8) & 0xF) - 8;
  int b7 = ((vh >> 12) & 0xF) - 8;
  hi = (b4 & 0xFF) | ((b5 & 0xFF) << 8) | ((b6 & 0xFF) << 16) | ((b7 & 0xFF) << 24);
}

template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_prod(
    const int* __restrict__ A,        // [M, K/4] знакові байти a (4 на int32)
    const int* __restrict__ B,        // [N, K/8] беззнакові напівбайти q (8 на int32)
    const float* __restrict__ Sa_sum, // [M, NG] знакові суми a по групі
    const float* __restrict__ Sw,     // [N, NG] масштаби ваг
    const float* __restrict__ Zp,     // [N, NG] нульові точки (0..15)
    const float* __restrict__ Sa,     // [M, NG] масштаб активацій на групу
    float* __restrict__ C, int M, int N, int K) {
  const int K4 = K / 4, K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN); // 256
  const int STM = BM / TM, STN = BN / TN; // 16, 16
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  const int W4_A = WORDS / 4;        // 32 / 4 = 8
  const int rowsA = nthr / W4_A;     // 256 / 8 = 32
  const int PA_4 = (BM * WORDS / nthr) / 4; // (128 * 32 / 256) / 4 = 4

  const int W4_B = WORDS_RAW / 4;    // 16 / 4 = 4
  const int rowsB = nthr / W4_B;     // 256 / 4 = 64
  const int PB_4 = (BN * WORDS_RAW / nthr) / 4; // (64 * 16 / 256) / 4 = 1

  // Буфери LDS:
  // sA: одинарний буфер активацій [128][34] * 4 Б = 17 408 Б
  // sB: подвійний буфер розпакованих ваг 2 * [64][34] * 4 Б = 17 408 Б
  // Масштаби та суми: sSa_sum (512 Б), sSa (512 Б), sSw (512 Б), sZps (512 Б) = 2 048 Б
  // Загальний бюджет LDS: 17408 + 17408 + 2048 = 36 864 Б (36.0 КБ) -> 1 блок на CU (LDS CU = 64 КБ).
  __shared__ __align__(8) int sA[BM][WORDS + 2];
  __shared__ __align__(8) int sB[2][BN][WORDS + 2];
  __shared__ float sSa_sum[BM], sSa[BM];
  __shared__ float sSw[2][BN], sZps[2][BN];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  int4v ra[PA_4 > 0 ? PA_4 : 1], rb[PB_4 > 0 ? PB_4 : 1];

  // Початкове завантаження групи 0 в LDS
#pragma unroll
  for (int k = 0; k < PA_4; ++k) {
    int r = tid / W4_A + k * rowsA, w = (tid % W4_A) * 4, m = m0 + r;
    int4v v = (m < M) ? *(const int4v*)(&A[(long)m * K4 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sA[r][w + e] = v[e];
  }

#pragma unroll
  for (int k = 0; k < PB_4; ++k) {
    int r = tid / W4_B + k * rowsB, w_raw = (tid % W4_B) * 4, n = n0 + r;
    int4v v = (n < N) ? *(const int4v*)(&B[(long)n * K8 + w_raw]) : (int4v)(0);
    int w_lds = w_raw * 2;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int lo, hi;
      unpack_nibbles(v[e], lo, hi);
      sB[0][r][w_lds + 2 * e] = lo;
      sB[0][r][w_lds + 2 * e + 1] = hi;
    }
  }

  if (tid < BM) {
    int m = m0 + tid;
    sSa_sum[tid] = (m < M) ? Sa_sum[(long)m * NG] : 0.f;
    sSa[tid] = (m < M) ? Sa[(long)m * NG] : 0.f;
  }
  if (tid < BN) {
    int n = n0 + tid;
    sSw[0][tid] = (n < N) ? Sw[(long)n * NG] : 0.f;
    sZps[0][tid] = (n < N) ? (Zp[(long)n * NG] - 8.f) : 0.f;
  }
  __syncthreads();

  for (int g = 0; g < NG; ++g) {
    const int cur = g & 1, nxt = cur ^ 1;

    // Випереджальне читання (prefetch) наступної групи g + 1 у регістри
    if (g + 1 < NG) {
      __builtin_amdgcn_s_setprio(1);
#pragma unroll
      for (int k = 0; k < PA_4; ++k) {
        int r = tid / W4_A + k * rowsA, w = (tid % W4_A) * 4, m = m0 + r;
        ra[k] = (m < M) ? *(const int4v*)(&A[(long)m * K4 + (g + 1) * WORDS + w])
                        : (int4v)(0);
      }
#pragma unroll
      for (int k = 0; k < PB_4; ++k) {
        int r = tid / W4_B + k * rowsB, w_raw = (tid % W4_B) * 4, n = n0 + r;
        rb[k] = (n < N) ? *(const int4v*)(&B[(long)n * K8 + (g + 1) * WORDS_RAW + w_raw])
                        : (int4v)(0);
      }
      __builtin_amdgcn_s_setprio(0);
    }

    // Обчислення dot product для поточної групи g
    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0;

#pragma unroll 1
    for (int w = 0; w < WORDS; w += 2) {
      int2v av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = *(const int2v*)(&sA[tm + i * STM][w]);
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = *(const int2v*)(&sB[cur][tn + j * STN][w]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] = __builtin_amdgcn_sdot4(av[i][0], bv[j][0], acc[i][j], false);
          acc[i][j] = __builtin_amdgcn_sdot4(av[i][1], bv[j][1], acc[i][j], false);
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
      float sa_sum = sSa_sum[tm + i * STM];
      float sag = sSa[tm + i * STM];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int c = tn + j * STN;
        float v = (float)acc[i][j] - sZps[cur][c] * sa_sum;
        fout[i][j] += (sSw[cur][c] * sag) * v;
      }
    }

    // Запис випереджально зчитаних даних наступної групи g + 1 в LDS
    if (g + 1 < NG) {
      __syncthreads();
#pragma unroll
      for (int k = 0; k < PA_4; ++k) {
        int r = tid / W4_A + k * rowsA, w = (tid % W4_A) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) sA[r][w + e] = ra[k][e];
      }
#pragma unroll
      for (int k = 0; k < PB_4; ++k) {
        int r = tid / W4_B + k * rowsB, w_raw = (tid % W4_B) * 4;
        int w_lds = w_raw * 2;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int lo, hi;
          unpack_nibbles(rb[k][e], lo, hi);
          sB[nxt][r][w_lds + 2 * e] = lo;
          sB[nxt][r][w_lds + 2 * e + 1] = hi;
        }
      }
      if (tid < BM) {
        int m = m0 + tid;
        sSa_sum[tid] = (m < M) ? Sa_sum[(long)m * NG + g + 1] : 0.f;
        sSa[tid] = (m < M) ? Sa[(long)m * NG + g + 1] : 0.f;
      }
      if (tid < BN) {
        int n = n0 + tid;
        sSw[nxt][tid] = (n < N) ? Sw[(long)n * NG + g + 1] : 0.f;
        sZps[nxt][tid] = (n < N) ? (Zp[(long)n * NG + g + 1] - 8.f) : 0.f;
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int m = m0 + tm + i * STM;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
  }
}

static inline int sbyte(int word, int k) { return (signed char)((word >> (8 * k)) & 0xFF); }
static inline int nib(int word, int k) { return (word >> (4 * k)) & 0xF; }

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int K = argc > 2 ? atoi(argv[2]) : 5120;
  int N = argc > 3 ? atoi(argv[3]) : 17408;
  int K4 = K / 4, K8 = K / 8, NG = K / GS;
  hipDeviceProp_t p; (void)hipGetDeviceProperties(&p, 0);
  printf("пристрій: %s   M=%d K=%d N=%d (W4A8: знакові байти x знакові напівбайти + нульові точки)\n",
         p.name, M, K, N);

  std::vector<int> hA((long)M * K4), hB((long)N * K8);
  std::vector<float> hSa_sum((long)M * NG), hSw((long)N * NG), hZp((long)N * NG),
      hSa((long)M * NG);
  srand(7);
  for (auto& v : hA) {
    int b0 = (rand() % 255) - 127;
    int b1 = (rand() % 255) - 127;
    int b2 = (rand() % 255) - 127;
    int b3 = (rand() % 255) - 127;
    v = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
  }
  for (auto& v : hB) {
    v = 0;
    for (int k = 0; k < 8; ++k) v |= (rand() % 16) << (4 * k);
  }
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < NG; ++g)
      hSa[(long)m * NG + g] = 0.002f + (rand() % 50) * 1e-5f;
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < NG; ++g) {
      hSw[(long)n * NG + g] = 0.004f + (rand() % 50) * 1e-5f;
      hZp[(long)n * NG + g] = (float)(rand() % 16);
    }

  // Похідні суми: Sa_sum по (рядок, група)
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < NG; ++g) {
      long s = 0;
      for (int w = 0; w < WORDS; ++w)
        for (int k = 0; k < 4; ++k)
          s += sbyte(hA[(long)m * K4 + g * WORDS + w], k);
      hSa_sum[(long)m * NG + g] = (float)s;
    }

  int *dA, *dB; float *dSa_sum, *dSw, *dZp, *dSa, *dC;
  hipMalloc(&dA, hA.size() * 4);           hipMalloc(&dB, hB.size() * 4);
  hipMalloc(&dSa_sum, hSa_sum.size() * 4); hipMalloc(&dSw, hSw.size() * 4);
  hipMalloc(&dZp, hZp.size() * 4);         hipMalloc(&dSa, hSa.size() * 4);
  hipMalloc(&dC, (long)M * N * 4);
  hipMemcpy(dA, hA.data(), hA.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB.data(), hB.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dSa_sum, hSa_sum.data(), hSa_sum.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dSw, hSw.data(), hSw.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dZp, hZp.data(), hZp.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dSa, hSa.data(), hSa.size() * 4, hipMemcpyHostToDevice);

  dim3 blk(128 * 64 / 32), grd((N + 63) / 64, (M + 127) / 128);
  for (int i = 0; i < 5; ++i)
    gemm_prod<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dSa_sum, dSw, dZp, dSa, dC, M, N, K);
  hipDeviceSynchronize();
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0);
  for (int r = 0; r < 20; ++r)
    gemm_prod<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dSa_sum, dSw, dZp, dSa, dC, M, N, K);
  hipEventRecord(e1); hipEventSynchronize(e1);
  float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 20;
  printf("  W4A8 (знаковий, групові масштаби + одинарний sA): %7.3f мс  %6.2f ТОпс\n", ms,
         2.0 * M * N * K / (ms / 1e3) / 1e12);

  std::vector<float> hC((long)M * N);
  hipMemcpy(hC.data(), dC, (long)M * N * 4, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int t = 0; t < 12; ++t) {
    int m = rand() % M, n = rand() % N;
    double ref = 0;
    for (int g = 0; g < NG; ++g) {
      double s = 0;
      for (int w = 0; w < WORDS_RAW; ++w) {
        int wA0 = g * WORDS + 2 * w;
        int wA1 = g * WORDS + 2 * w + 1;
        int wB = g * WORDS_RAW + w;
        int vA0 = hA[(long)m * K4 + wA0];
        int vA1 = hA[(long)m * K4 + wA1];
        int vB  = hB[(long)n * K8 + wB];
        for (int k = 0; k < 4; ++k) {
          double a = sbyte(vA0, k);
          double q = nib(vB, k);
          s += a * (q - hZp[(long)n * NG + g]);
        }
        for (int k = 0; k < 4; ++k) {
          double a = sbyte(vA1, k);
          double q = nib(vB, k + 4);
          s += a * (q - hZp[(long)n * NG + g]);
        }
      }
      ref += s * hSw[(long)n * NG + g] * hSa[(long)m * NG + g];
    }
    double got = hC[(long)m * N + n];
    if (fabs(got - ref) / (fabs(ref) + 1e-3) > 1e-4) {
      ++bad;
      printf("  РОЗБІЖНІСТЬ m=%d n=%d: %.5f проти %.5f\n", m, n, got, ref);
    }
  }
  printf("  правильність: %s\n", bad ? "ПОМИЛКА" : "збігається з прямим рахунком");
  return bad ? 1 : 0;
}
