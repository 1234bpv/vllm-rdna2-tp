// Прототип W4A4 GEMM на gfx1030: обидва операнди 4-бітні, добуток через
// v_dot8_i32_i4 (вбудоване __builtin_amdgcn_sdot8, 8 множень за такт).
//
// Розкладка операндів навмисне така сама, як у бойовому шляху AWQ: ваги вже
// лежать [N, K/8] упаковані по 8 напівбайтів у int32, тобто вздовж K — рівно
// те, що потрібно добутку. Активації пакуються так само [M, K/8].
//
// Масштаби враховано чесно: int32 накопичується в межах ГРУПИ (128 значень
// уздовж K), потім переводиться у float і множиться на масштаб рядка й
// групи. Ігнорувати це означало б завищити швидкість.
//
// ⚠️ Імена в коді латиницею.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define GS 128        // розмір групи вздовж K
#define WORDS (GS/8)  // int32-слів на групу

#include <cmath>
typedef int int4v __attribute__((ext_vector_type(4)));
typedef int int2v __attribute__((ext_vector_type(2)));

template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void gemm_i4_v13_body(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int PA = BM * 2 * WORDS / nthr;   // скільки слів A на нитку
  const int PB = BN * 2 * WORDS / nthr;   // скільки слів B на нитку

  __shared__ __align__(8) int sA[2][BM][2 * WORDS + 2];
  __shared__ __align__(8) int sB[2][BN][2 * WORDS + 2];
  __shared__ float sAs[2][2][BM];   // [буфер][група в парі][рядок]
  __shared__ float sBs[2][2][BN];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  // ⚠️ Глобальні читання ШИРОКІ: одне 128-бітне замість чотирьох окремих.
  // Кожна нитка бере чотири СУСІДНІ слова одного рядка, тож чотири нитки
  // вкривають рядок суцільно. У LDS запис лишається чотирма словами: крок
  // рядка там непарний (17), і 128-бітний запис туди не вирівняти, а міняти
  // крок не можна — на ньому тримається безконфліктність читань у циклі.
  const int RPT = BM / (nthr / (WORDS / 4));  // рядків на нитку за прохід
  int4v ra[(PA / 4) > 0 ? (PA / 4) : 1], rb[(PB / 4) > 0 ? (PB / 4) : 1];
  (void)RPT;

  // Перша плитка.
  const int W4 = 2 * WORDS / 4;          // 128-бітних порцій у рядку
  const int rowsA = nthr / W4;       // скільки рядків бере блок за прохід
#pragma unroll
  for (int k = 0; k < PA / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
    int m = m0 + r;
    int4v v = (m < M) ? *(const int4v*)(&A[(long)m * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sA[0][r][w + e] = v[e];
  }
#pragma unroll
  for (int k = 0; k < PB / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
    int n = n0 + r;
    int4v v = (n < N) ? *(const int4v*)(&B[(long)n * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sB[0][r][w + e] = v[e];
  }
  for (int t = tid; t < 2 * BM; t += nthr) {
    int gg = t / BM, r = t % BM, m = m0 + r;
    sAs[0][gg][r] = (m < M) ? As[(long)m * NG + gg] : 0.f;
  }
  for (int t = tid; t < 2 * BN; t += nthr) {
    int gg = t / BN, r = t % BN, n = n0 + r;
    sBs[0][gg][r] = (n < N) ? Bs[(long)n * NG + gg] : 0.f;
  }
  __syncthreads();

  for (int g = 0; g < NG; g += 2) {
    // ⚠️ g росте НА ДВА, тож `g & 1` було б завжди нулем і ядро читало б
    // застарілий буфер. Чергувати треба за номером ІТЕРАЦІЇ.
    const int cur = (g >> 1) & 1, nxt = cur ^ 1;
    // Замовляємо наступну плитку ДО рахунку — читання летять паралельно.
    if (g + 2 < NG) {
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int m = m0 + r;
        ra[k] = (m < M) ? *(const int4v*)(&A[(long)m * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int n = n0 + r;
        rb[k] = (n < N) ? *(const int4v*)(&B[(long)n * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
    }

#pragma unroll 1
   for (int gg = 0; gg < 2 && g + gg < NG; ++gg) {
    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0;
#pragma unroll 1
    for (int w = gg * WORDS; w < (gg + 1) * WORDS; w += 2) {
      int2v av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        av[i] = *(const int2v*)(&sA[cur][tm + i * STM][w]);
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bv[j] = *(const int2v*)(&sB[cur][tn + j * STN][w]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] = __builtin_amdgcn_sdot8(av[i][0], bv[j][0], acc[i][j], false);
          acc[i][j] = __builtin_amdgcn_sdot8(av[i][1], bv[j][1], acc[i][j], false);
        }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      int m = m0 + tm + i * STM;
      float as = sAs[cur][gg][tm + i * STM];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int n = n0 + tn + j * STN;
        float bs = sBs[cur][gg][tn + j * STN];
        fout[i][j] += (float)acc[i][j] * as * bs;
      }
    }
   }

    if (g + 2 < NG) {
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) sA[nxt][r][w + e] = ra[k][e];
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) sB[nxt][r][w + e] = rb[k][e];
      }
      // Масштаби наступної пари груп — теж у LDS. Без цього рядка ядро читає
      // масштаби ПЕРШОЇ пари для всіх груп і рахує неправильно, лишаючись
      // швидким: саме так і сталося при першій спробі.
      for (int t = tid; t < 2 * BM; t += nthr) {
        int gg2 = t / BM, r = t % BM, m = m0 + r;
        sAs[nxt][gg2][r] = (m < M) ? As[(long)m * NG + g + 2 + gg2] : 0.f;
      }
      for (int t = tid; t < 2 * BN; t += nthr) {
        int gg2 = t / BN, r = t % BN, n = n0 + r;
        sBs[nxt][gg2][r] = (n < N) ? Bs[(long)n * NG + g + 2 + gg2] : 0.f;
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}

// ── Варіант 1: gemm_lds_a ──────────────────────────────────────────────────
// Прибираємо подвійну буферизацію лише для sA (і масштабів sAs), лишаючи її для sB.
// Потрібен додатковий бар'єр перед записом нової плитки в sA, оскільки старі дані
// ще можуть читатися нитками в поточному обчислювальному циклі.
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void gemm_lds_a(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int PA = BM * 2 * WORDS / nthr;   // скільки слів A на нитку
  const int PB = BN * 2 * WORDS / nthr;   // скільки слів B на нитку

  __shared__ __align__(8) int sA[BM][2 * WORDS + 2];
  __shared__ __align__(8) int sB[2][BN][2 * WORDS + 2];
  __shared__ float sAs[2][BM];   // [група в парі][рядок]
  __shared__ float sBs[2][2][BN];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  const int RPT = BM / (nthr / (WORDS / 4));
  int4v ra[(PA / 4) > 0 ? (PA / 4) : 1], rb[(PB / 4) > 0 ? (PB / 4) : 1];
  (void)RPT;

  // Перша плитка.
  const int W4 = 2 * WORDS / 4;
  const int rowsA = nthr / W4;
#pragma unroll
  for (int k = 0; k < PA / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
    int m = m0 + r;
    int4v v = (m < M) ? *(const int4v*)(&A[(long)m * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sA[r][w + e] = v[e];
  }
#pragma unroll
  for (int k = 0; k < PB / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
    int n = n0 + r;
    int4v v = (n < N) ? *(const int4v*)(&B[(long)n * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sB[0][r][w + e] = v[e];
  }
  for (int t = tid; t < 2 * BM; t += nthr) {
    int gg = t / BM, r = t % BM, m = m0 + r;
    sAs[gg][r] = (m < M) ? As[(long)m * NG + gg] : 0.f;
  }
  for (int t = tid; t < 2 * BN; t += nthr) {
    int gg = t / BN, r = t % BN, n = n0 + r;
    sBs[0][gg][r] = (n < N) ? Bs[(long)n * NG + gg] : 0.f;
  }
  __syncthreads();

  for (int g = 0; g < NG; g += 2) {
    const int cur = (g >> 1) & 1, nxt = cur ^ 1;
    if (g + 2 < NG) {
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int m = m0 + r;
        ra[k] = (m < M) ? *(const int4v*)(&A[(long)m * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int n = n0 + r;
        rb[k] = (n < N) ? *(const int4v*)(&B[(long)n * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
    }

#pragma unroll 1
    for (int gg = 0; gg < 2 && g + gg < NG; ++gg) {
      int acc[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;
#pragma unroll 1
      for (int w = gg * WORDS; w < (gg + 1) * WORDS; w += 2) {
        int2v av[TM], bv[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
          av[i] = *(const int2v*)(&sA[tm + i * STM][w]);
#pragma unroll
        for (int j = 0; j < TN; ++j)
          bv[j] = *(const int2v*)(&sB[cur][tn + j * STN][w]);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = __builtin_amdgcn_sdot8(av[i][0], bv[j][0], acc[i][j], false);
            acc[i][j] = __builtin_amdgcn_sdot8(av[i][1], bv[j][1], acc[i][j], false);
          }
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int m = m0 + tm + i * STM;
        float as = sAs[gg][tm + i * STM];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int n = n0 + tn + j * STN;
          float bs = sBs[cur][gg][tn + j * STN];
          fout[i][j] += (float)acc[i][j] * as * bs;
        }
      }
    }

    if (g + 2 < NG) {
      __syncthreads(); // Бар'єр 1: гарантуємо, що всі нитки завершили читання з sA та sAs
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) sA[r][w + e] = ra[k][e];
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) sB[nxt][r][w + e] = rb[k][e];
      }
      for (int t = tid; t < 2 * BM; t += nthr) {
        int gg2 = t / BM, r = t % BM, m = m0 + r;
        sAs[gg2][r] = (m < M) ? As[(long)m * NG + g + 2 + gg2] : 0.f;
      }
      for (int t = tid; t < 2 * BN; t += nthr) {
        int gg2 = t / BN, r = t % BN, n = n0 + r;
        sBs[nxt][gg2][r] = (n < N) ? Bs[(long)n * NG + g + 2 + gg2] : 0.f;
      }
      __syncthreads(); // Бар'єр 2: гарантуємо запис перед наступною ітерацією
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}

// ── Варіант 2: gemm_lds_nopad ──────────────────────────────────────────────
// Лишаємо подвійну буферизацію, але прибираємо доповнення рядків (+2) у LDS.
// Рядок має довжину рівно 2*WORDS (32 слова = 128 байтів), що викликає конфлікти банків.
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void gemm_lds_nopad(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int PA = BM * 2 * WORDS / nthr;   // скільки слів A на нитку
  const int PB = BN * 2 * WORDS / nthr;   // скільки слів B на нитку

  __shared__ __align__(8) int sA[2][BM][2 * WORDS];
  __shared__ __align__(8) int sB[2][BN][2 * WORDS];
  __shared__ float sAs[2][2][BM];   // [буфер][група в парі][рядок]
  __shared__ float sBs[2][2][BN];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  const int RPT = BM / (nthr / (WORDS / 4));
  int4v ra[(PA / 4) > 0 ? (PA / 4) : 1], rb[(PB / 4) > 0 ? (PB / 4) : 1];
  (void)RPT;

  // Перша плитка.
  const int W4 = 2 * WORDS / 4;
  const int rowsA = nthr / W4;
#pragma unroll
  for (int k = 0; k < PA / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
    int m = m0 + r;
    int4v v = (m < M) ? *(const int4v*)(&A[(long)m * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sA[0][r][w + e] = v[e];
  }
#pragma unroll
  for (int k = 0; k < PB / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
    int n = n0 + r;
    int4v v = (n < N) ? *(const int4v*)(&B[(long)n * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sB[0][r][w + e] = v[e];
  }
  for (int t = tid; t < 2 * BM; t += nthr) {
    int gg = t / BM, r = t % BM, m = m0 + r;
    sAs[0][gg][r] = (m < M) ? As[(long)m * NG + gg] : 0.f;
  }
  for (int t = tid; t < 2 * BN; t += nthr) {
    int gg = t / BN, r = t % BN, n = n0 + r;
    sBs[0][gg][r] = (n < N) ? Bs[(long)n * NG + gg] : 0.f;
  }
  __syncthreads();

  for (int g = 0; g < NG; g += 2) {
    const int cur = (g >> 1) & 1, nxt = cur ^ 1;
    if (g + 2 < NG) {
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int m = m0 + r;
        ra[k] = (m < M) ? *(const int4v*)(&A[(long)m * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int n = n0 + r;
        rb[k] = (n < N) ? *(const int4v*)(&B[(long)n * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
    }

#pragma unroll 1
    for (int gg = 0; gg < 2 && g + gg < NG; ++gg) {
      int acc[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;
#pragma unroll 1
      for (int w = gg * WORDS; w < (gg + 1) * WORDS; w += 2) {
        int2v av[TM], bv[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
          av[i] = *(const int2v*)(&sA[cur][tm + i * STM][w]);
#pragma unroll
        for (int j = 0; j < TN; ++j)
          bv[j] = *(const int2v*)(&sB[cur][tn + j * STN][w]);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = __builtin_amdgcn_sdot8(av[i][0], bv[j][0], acc[i][j], false);
            acc[i][j] = __builtin_amdgcn_sdot8(av[i][1], bv[j][1], acc[i][j], false);
          }
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int m = m0 + tm + i * STM;
        float as = sAs[cur][gg][tm + i * STM];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int n = n0 + tn + j * STN;
          float bs = sBs[cur][gg][tn + j * STN];
          fout[i][j] += (float)acc[i][j] * as * bs;
        }
      }
    }

    if (g + 2 < NG) {
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) sA[nxt][r][w + e] = ra[k][e];
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) sB[nxt][r][w + e] = rb[k][e];
      }
      for (int t = tid; t < 2 * BM; t += nthr) {
        int gg2 = t / BM, r = t % BM, m = m0 + r;
        sAs[nxt][gg2][r] = (m < M) ? As[(long)m * NG + g + 2 + gg2] : 0.f;
      }
      for (int t = tid; t < 2 * BN; t += nthr) {
        int gg2 = t / BN, r = t % BN, n = n0 + r;
        sBs[nxt][gg2][r] = (n < N) ? Bs[(long)n * NG + g + 2 + gg2] : 0.f;
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}

// ── Варіант 3: gemm_lds_single ─────────────────────────────────────────────
// Прибираємо подвійну буферизацію для обох sA та sB (і відповідно їх масштабів sAs, sBs).
// Потрібен бар'єр перед заповненням нової плитки і бар'єр після заповнення.
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void gemm_lds_single(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int PA = BM * 2 * WORDS / nthr;   // скільки слів A на нитку
  const int PB = BN * 2 * WORDS / nthr;   // скільки слів B на нитку

  __shared__ __align__(8) int sA[BM][2 * WORDS + 2];
  __shared__ __align__(8) int sB[BN][2 * WORDS + 2];
  __shared__ float sAs[2][BM];   // [група в парі][рядок]
  __shared__ float sBs[2][BN];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  const int RPT = BM / (nthr / (WORDS / 4));
  int4v ra[(PA / 4) > 0 ? (PA / 4) : 1], rb[(PB / 4) > 0 ? (PB / 4) : 1];
  (void)RPT;

  // Перша плитка.
  const int W4 = 2 * WORDS / 4;
  const int rowsA = nthr / W4;
#pragma unroll
  for (int k = 0; k < PA / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
    int m = m0 + r;
    int4v v = (m < M) ? *(const int4v*)(&A[(long)m * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sA[r][w + e] = v[e];
  }
#pragma unroll
  for (int k = 0; k < PB / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
    int n = n0 + r;
    int4v v = (n < N) ? *(const int4v*)(&B[(long)n * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sB[r][w + e] = v[e];
  }
  for (int t = tid; t < 2 * BM; t += nthr) {
    int gg = t / BM, r = t % BM, m = m0 + r;
    sAs[gg][r] = (m < M) ? As[(long)m * NG + gg] : 0.f;
  }
  for (int t = tid; t < 2 * BN; t += nthr) {
    int gg = t / BN, r = t % BN, n = n0 + r;
    sBs[gg][r] = (n < N) ? Bs[(long)n * NG + gg] : 0.f;
  }
  __syncthreads();

  for (int g = 0; g < NG; g += 2) {
    if (g + 2 < NG) {
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int m = m0 + r;
        ra[k] = (m < M) ? *(const int4v*)(&A[(long)m * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int n = n0 + r;
        rb[k] = (n < N) ? *(const int4v*)(&B[(long)n * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
    }

#pragma unroll 1
    for (int gg = 0; gg < 2 && g + gg < NG; ++gg) {
      int acc[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;
#pragma unroll 1
      for (int w = gg * WORDS; w < (gg + 1) * WORDS; w += 2) {
        int2v av[TM], bv[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
          av[i] = *(const int2v*)(&sA[tm + i * STM][w]);
#pragma unroll
        for (int j = 0; j < TN; ++j)
          bv[j] = *(const int2v*)(&sB[tn + j * STN][w]);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = __builtin_amdgcn_sdot8(av[i][0], bv[j][0], acc[i][j], false);
            acc[i][j] = __builtin_amdgcn_sdot8(av[i][1], bv[j][1], acc[i][j], false);
          }
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int m = m0 + tm + i * STM;
        float as = sAs[gg][tm + i * STM];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int n = n0 + tn + j * STN;
          float bs = sBs[gg][tn + j * STN];
          fout[i][j] += (float)acc[i][j] * as * bs;
        }
      }
    }

    if (g + 2 < NG) {
      __syncthreads(); // Бар'єр 1: гарантуємо, що всі нитки завершили читання з sA, sB, sAs, sBs
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) sA[r][w + e] = ra[k][e];
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) sB[r][w + e] = rb[k][e];
      }
      for (int t = tid; t < 2 * BM; t += nthr) {
        int gg2 = t / BM, r = t % BM, m = m0 + r;
        sAs[gg2][r] = (m < M) ? As[(long)m * NG + g + 2 + gg2] : 0.f;
      }
      for (int t = tid; t < 2 * BN; t += nthr) {
        int gg2 = t / BN, r = t % BN, n = n0 + r;
        sBs[gg2][r] = (n < N) ? Bs[(long)n * NG + g + 2 + gg2] : 0.f;
      }
      __syncthreads(); // Бар'єр 2: гарантуємо завершення запису перед наступною ітерацією
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}

// ── перевірка зайнятості на теперішній межі (52 ТОпс) ───────────────────────
// Негативний результат «більше хвиль не допомагає» здобуто на межі 37 ТОпс,
// до перенесення масштабів груп у LDS. Межа зсунулася — переміряємо.
// Зайнятість обмежуємо атрибутом amdgpu_waves_per_eu(NW, NW): компілятор
// зобов'язаний укластися у стільки VGPR, щоб NW хвиль на SIMD влізло.
#define KERNEL_OCC(NW)                                                        \
template <int BM, int BN, int TM, int TN>                                     \
__global__ __launch_bounds__(256)                                             \
__attribute__((amdgpu_waves_per_eu(NW, NW)))                                  \
void gemm_occ##NW(const int* __restrict__ A, const int* __restrict__ B,       \
                  const float* __restrict__ As, const float* __restrict__ Bs, \
                  float* __restrict__ C, int M, int N, int K) {               \
  gemm_i4_v13_body<BM, BN, TM, TN>(A, B, As, Bs, C, M, N, K);                 \
}
KERNEL_OCC(1) KERNEL_OCC(2) KERNEL_OCC(3) KERNEL_OCC(4)
KERNEL_OCC(6) KERNEL_OCC(8)

// Обгортки для нових варіантів LDS
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256)
void gemm_k_lds_a(const int* __restrict__ A, const int* __restrict__ B,
                  const float* __restrict__ As, const float* __restrict__ Bs,
                  float* __restrict__ C, int M, int N, int K) {
  gemm_lds_a<BM, BN, TM, TN>(A, B, As, Bs, C, M, N, K);
}

template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256)
void gemm_k_lds_nopad(const int* __restrict__ A, const int* __restrict__ B,
                      const float* __restrict__ As, const float* __restrict__ Bs,
                      float* __restrict__ C, int M, int N, int K) {
  gemm_lds_nopad<BM, BN, TM, TN>(A, B, As, Bs, C, M, N, K);
}

template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256)
void gemm_k_lds_single(const int* __restrict__ A, const int* __restrict__ B,
                       const float* __restrict__ As, const float* __restrict__ Bs,
                       float* __restrict__ C, int M, int N, int K) {
  gemm_lds_single<BM, BN, TM, TN>(A, B, As, Bs, C, M, N, K);
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int K = argc > 2 ? atoi(argv[2]) : 5120;
  int N = argc > 3 ? atoi(argv[3]) : 17408;
  int K8 = K / 8, NG = K / GS;
  hipDeviceProp_t p; (void)hipGetDeviceProperties(&p, 0);
  printf("пристрій: %s   M=%d K=%d N=%d\n", p.name, M, K, N);

  std::vector<int> hA((long)M * K8), hB((long)N * K8);
  std::vector<float> hAs((long)M * NG), hBs((long)N * NG);
  srand(1);
  for (auto& v : hA) v = rand();
  for (auto& v : hB) v = rand();
  for (auto& v : hAs) v = 0.01f + (rand() % 100) * 1e-4f;
  for (auto& v : hBs) v = 0.01f + (rand() % 100) * 1e-4f;

  int *dA, *dB; float *dAs, *dBs, *dC;
  hipMalloc(&dA, hA.size() * 4);   hipMalloc(&dB, hB.size() * 4);
  hipMalloc(&dAs, hAs.size() * 4); hipMalloc(&dBs, hBs.size() * 4);
  hipMalloc(&dC, (long)M * N * 4);
  hipMemcpy(dA, hA.data(), hA.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB.data(), hB.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dAs, hAs.data(), hAs.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dBs, hBs.data(), hBs.size() * 4, hipMemcpyHostToDevice);

  // еталон для перевірки правильності: 12 випадкових елементів
  std::vector<float> hC((long)M * N);
  auto perevir = [&](const char* nazva) {
    hipMemcpy(hC.data(), dC, (long)M * N * 4, hipMemcpyDeviceToHost);
    double najg = 0;
    for (int t = 0; t < 12; ++t) {
      int i = rand() % M, j = rand() % N; double acc = 0;
      for (int g = 0; g < NG; ++g) {
        long s = 0;
        for (int w = 0; w < WORDS; ++w) {
          unsigned a = hA[(long)i * K8 + g * WORDS + w];
          unsigned b = hB[(long)j * K8 + g * WORDS + w];
          // ⚠️ ЗНАКОВІ півбайти: v13 — прототип на v_dot8_i32_i4. Беззнаковий
          // еталон давав розбіжність 0,97, і це була вада приладу, не ядра.
          for (int n = 0; n < 8; ++n) {
            int av = (a >> (4 * n)) & 15, bv = (b >> (4 * n)) & 15;
            s += (long)((av & 8) ? av - 16 : av) * (long)((bv & 8) ? bv - 16 : bv);
          }
        }
        acc += (double)s * hAs[(long)i * NG + g] * hBs[(long)j * NG + g];
      }
      double e = fabs(acc - hC[(long)i * N + j]) / (fabs(acc) + 1e-9);
      if (e > najg) najg = e;
    }
    if (najg > 1e-4) printf("   ⚠️ %s: РОЗБІЖНІСТЬ %.3e\n", nazva, najg);
    return najg;
  };

  dim3 blk(256), grd((N + 63) / 64, (M + 127) / 128);
#define PROBA_OCC(NW)                                                         \
  {                                                                           \
    hipFuncAttributes fa;                                                     \
    hipFuncGetAttributes(&fa, (const void*)gemm_occ##NW<128, 64, 8, 4>);      \
    int blocks = 0;                                                           \
    hipOccupancyMaxActiveBlocksPerMultiprocessor(                             \
        &blocks, (const void*)gemm_occ##NW<128, 64, 8, 4>, 256, 0);           \
    for (int i = 0; i < 5; ++i)                                               \
      gemm_occ##NW<128, 64, 8, 4><<<grd, blk>>>(dA,dB,dAs,dBs,dC,M,N,K);      \
    hipDeviceSynchronize();                                                   \
    double naj = perevir("хвиль=" #NW);                                       \
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);              \
    const int rep = 20; hipEventRecord(e0);                                   \
    for (int r = 0; r < rep; ++r)                                             \
      gemm_occ##NW<128, 64, 8, 4><<<grd, blk>>>(dA,dB,dAs,dBs,dC,M,N,K);      \
    hipEventRecord(e1); hipEventSynchronize(e1);                              \
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;                \
    printf("  просив %-2d  VGPR %3d  LDS %5zu Б  блоків/CU %d  хвиль/SIMD %.1f"\
           " : %7.3f мс  %6.2f ТОпс   похибка %.1e\n", NW, fa.numRegs,        \
           fa.sharedSizeBytes, blocks, blocks * 8 / 4.0, ms,                   \
           2.0 * M * N * K / (ms / 1e3) / 1e12, naj);                         \
  }
  PROBA_OCC(1) PROBA_OCC(2) PROBA_OCC(3) PROBA_OCC(4) PROBA_OCC(6) PROBA_OCC(8)

#define PROBA_KERNEL(KFUNC, NAME)                                             \
  {                                                                           \
    hipFuncAttributes fa;                                                     \
    hipFuncGetAttributes(&fa, (const void*)KFUNC<128, 64, 8, 4>);             \
    int blocks = 0;                                                           \
    hipOccupancyMaxActiveBlocksPerMultiprocessor(                             \
        &blocks, (const void*)KFUNC<128, 64, 8, 4>, 256, 0);                  \
    for (int i = 0; i < 5; ++i)                                               \
      KFUNC<128, 64, 8, 4><<<grd, blk>>>(dA,dB,dAs,dBs,dC,M,N,K);             \
    hipDeviceSynchronize();                                                   \
    double naj = perevir(NAME);                                               \
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);              \
    const int rep = 20; hipEventRecord(e0);                                   \
    for (int r = 0; r < rep; ++r)                                             \
      KFUNC<128, 64, 8, 4><<<grd, blk>>>(dA,dB,dAs,dBs,dC,M,N,K);             \
    hipEventRecord(e1); hipEventSynchronize(e1);                              \
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;                \
    printf("  %-12s  VGPR %3d  LDS %5zu Б  блоків/CU %d  хвиль/SIMD %.1f"     \
           " : %7.3f мс  %6.2f ТОпс   похибка %.1e\n", NAME, fa.numRegs,      \
           fa.sharedSizeBytes, blocks, blocks * 8 / 4.0, ms,                   \
           2.0 * M * N * K / (ms / 1e3) / 1e12, naj);                         \
  }

  printf("\n--- Нові варіанти LDS ---\n");
  PROBA_KERNEL(gemm_k_lds_a, "lds_a")
  PROBA_KERNEL(gemm_k_lds_nopad, "lds_nopad")
  PROBA_KERNEL(gemm_k_lds_single, "lds_single")

  return 0;
}
