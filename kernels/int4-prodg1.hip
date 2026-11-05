// Ядро W4A4 у БОЙОВОМУ форматі vLLM: ваги asymmetric pack-quantized,
// напівбайти БЕЗЗНАКОВІ (0..15), нульові точки на групу.
//
// Математика. Активація x = (a' - 8)*sa, де a' беззнакове 0..15.
// Вага W = (q - zp)*sw. Тоді для групи:
//   S x*W = sa*sw*[ udot8(a', q) - zp*Sa' + (8*zp*G - 8*Sq) ]
// ⚠️ ВАРІАНТ ІЗ ГРУПОВИМИ МАСШТАБАМИ АКТИВАЦІЙ (sa на групу зі 128, а не на
// рядок). Причина: замір 01.09 показав, що масштаб на весь рядок дає
// перплексію 168,7 проти 6,66 — один канальний викид серед 5 120 значень
// піднімає sa так, що решта падає у перший-другий рівень із шістнадцяти.
// Підміна з масштабом на групу дала 8,15, тобто у двадцять разів краще.
//
// Наслідок для арифметики: sa тепер залежить від ГРУПИ, тож його не можна
// винести за цикл, і згорнути поправку в один вектор D[n] теж не можна. Замість
// D зберігаємо E[n][g] = 8*zp[n][g]*G - 8*Sq[n][g] (та сама форма, що у sw і
// zp), а накопичення стає
//   fout += sa[m][g] * sw[n][g] * (udot8(a',q) - zp[n][g]*Sa'[m][g] + E[n][g])
// і в кінці C = fout, без жодного множення. Ціна — два додаткові масиви в LDS
// (+3 072 Б) і одне множення на елемент на групу.
//
// ⚠️ Порядок напівбайтів усередині слова не має значення, доки він ОДНАКОВИЙ
// в активаціях і вагах: добуток підсумовує всі вісім позицій. Тому
// перемішування exllama, у якому лежать бойові ваги, переупаковувати НЕ треба
// — досить пакувати активації так само.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define GS 128
#define WORDS (GS / 8)
typedef int int2v __attribute__((ext_vector_type(2)));
typedef int int4v __attribute__((ext_vector_type(4)));

template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_prodg1(
    const int* __restrict__ A,     // [M, K/8] беззнакові напівбайти a'
    const int* __restrict__ B,     // [N, K/8] беззнакові напівбайти q
    const float* __restrict__ SAp, // [M, NG] суми a' по групі
    const float* __restrict__ Sw,  // [N, NG] масштаби ваг
    const float* __restrict__ Zp,  // [N, NG] нульові точки
    const float* __restrict__ E,   // [N, NG] поправка 8*zp*G - 8*Sq
    const float* __restrict__ Sa,  // [M, NG] масштаб активацій на групу
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int PA = BM * 2 * WORDS / nthr, PB = BN * 2 * WORDS / nthr;
  const int W4 = 2 * WORDS / 4, rowsA = nthr / W4;

  // ⚠️ sA і суми активацій — ОДИНАРНІ буфери, sB і масштаби ваг — подвійні.
  // Заміряно 01.09: одинарний sA дає 54,8 ТОпс проти 52,1 (+5,2%, розкид
  // 0,15%), хоч зайнятість НЕ змінюється — вона й далі 1 блок на CU. Виграш
  // не від зайнятості, а від простішої адресації: зникає доданок cur*BM*34
  // в кожному читанні sA (регістрів 146 замість 148).
  // ⚠️ Одинарними ОБИДВА робити НЕ можна: LDS падає до 27 648 Б і зайнятість
  // подвоюється до 4 хвиль/SIMD, але швидкість валиться до 32,8 ТОпс.
  // ⚠️ Доповнення рядків (+2) прибирати НЕ можна: без нього 11,2 ТОпс,
  // тобто конфлікти банків дорожчі за все інше разом.
  __shared__ __align__(8) int sA[BM][2 * WORDS + 2];
  __shared__ __align__(8) int sB[2][BN][2 * WORDS + 2];
  __shared__ float sSA[2][BM], sSa[2][BM];
  __shared__ float sSw[2][2][BN], sZp[2][2][BN], sE[2][2][BN];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  int4v ra[(PA / 4) > 0 ? (PA / 4) : 1], rb[(PB / 4) > 0 ? (PB / 4) : 1];

#pragma unroll
  for (int k = 0; k < PA / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4, m = m0 + r;
    int4v v = (m < M) ? *(const int4v*)(&A[(long)m * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sA[r][w + e] = v[e];
  }
#pragma unroll
  for (int k = 0; k < PB / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4, n = n0 + r;
    int4v v = (n < N) ? *(const int4v*)(&B[(long)n * K8 + w]) : (int4v)(0);
#pragma unroll
    for (int e = 0; e < 4; ++e) sB[0][r][w + e] = v[e];
  }
  for (int t = tid; t < 2 * BM; t += nthr) {
    int gg = t / BM, r = t % BM, m = m0 + r;
    sSA[gg][r] = (m < M) ? SAp[(long)m * NG + gg] : 0.f;
    sSa[gg][r] = (m < M) ? Sa[(long)m * NG + gg] : 0.f;
  }
  for (int t = tid; t < 2 * BN; t += nthr) {
    int gg = t / BN, r = t % BN, n = n0 + r;
    sSw[0][gg][r] = (n < N) ? Sw[(long)n * NG + gg] : 0.f;
    sZp[0][gg][r] = (n < N) ? Zp[(long)n * NG + gg] : 0.f;
    sE[0][gg][r] = (n < N) ? E[(long)n * NG + gg] : 0.f;
  }
  __syncthreads();

  for (int g = 0; g < NG; g += 2) {
    const int cur = (g >> 1) & 1, nxt = cur ^ 1;
    if (g + 2 < NG) {
      __builtin_amdgcn_s_setprio(1);
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4, m = m0 + r;
        ra[k] = (m < M) ? *(const int4v*)(&A[(long)m * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4, n = n0 + r;
        rb[k] = (n < N) ? *(const int4v*)(&B[(long)n * K8 + (g + 2) * WORDS + w])
                        : (int4v)(0);
      }
      __builtin_amdgcn_s_setprio(0);
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
        for (int i = 0; i < TM; ++i) av[i] = *(const int2v*)(&sA[tm + i * STM][w]);
#pragma unroll
        for (int j = 0; j < TN; ++j) bv[j] = *(const int2v*)(&sB[cur][tn + j * STN][w]);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = __builtin_amdgcn_udot8(av[i][0], bv[j][0], acc[i][j], false);
            acc[i][j] = __builtin_amdgcn_udot8(av[i][1], bv[j][1], acc[i][j], false);
          }
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        float sap = sSA[gg][tm + i * STM];
        float sag = sSa[gg][tm + i * STM];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int c = tn + j * STN;
          float v = (float)acc[i][j] - sZp[cur][gg][c] * sap + sE[cur][gg][c];
          fout[i][j] += (sSw[cur][gg][c] * sag) * v;
        }
      }
    }

    if (g + 2 < NG) {
      // Одинарний sA означає конфлікт «запис після читання»: швидші хвилі
      // затерли б плитку, поки повільніші її ще читають. Цей бар'єр і є вся
      // плата за відмову від подвійного буфера.
      __syncthreads();
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
        sSA[gg2][r] = (m < M) ? SAp[(long)m * NG + g + 2 + gg2] : 0.f;
        sSa[gg2][r] = (m < M) ? Sa[(long)m * NG + g + 2 + gg2] : 0.f;
      }
      for (int t = tid; t < 2 * BN; t += nthr) {
        int gg2 = t / BN, r = t % BN, n = n0 + r;
        sSw[nxt][gg2][r] = (n < N) ? Sw[(long)n * NG + g + 2 + gg2] : 0.f;
        sZp[nxt][gg2][r] = (n < N) ? Zp[(long)n * NG + g + 2 + gg2] : 0.f;
        sE[nxt][gg2][r] = (n < N) ? E[(long)n * NG + g + 2 + gg2] : 0.f;
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

static inline int nib(int word, int k) { return (word >> (4 * k)) & 0xF; }

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int K = argc > 2 ? atoi(argv[2]) : 5120;
  int N = argc > 3 ? atoi(argv[3]) : 17408;
  int K8 = K / 8, NG = K / GS;
  hipDeviceProp_t p; (void)hipGetDeviceProperties(&p, 0);
  printf("пристрій: %s   M=%d K=%d N=%d (бойовий формат: беззнакові напівбайти + нульові точки)\n",
         p.name, M, K, N);

  std::vector<int> hA((long)M * K8), hB((long)N * K8);
  std::vector<float> hSAp((long)M * NG), hSw((long)N * NG), hZp((long)N * NG),
      hE((long)N * NG), hSa((long)M * NG);
  srand(7);
  for (auto& v : hA) { v = 0; for (int k = 0; k < 8; ++k) v |= (rand() % 16) << (4 * k); }
  for (auto& v : hB) { v = 0; for (int k = 0; k < 8; ++k) v |= (rand() % 16) << (4 * k); }
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < NG; ++g)
      hSa[(long)m * NG + g] = 0.002f + (rand() % 50) * 1e-5f;
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < NG; ++g) {
      hSw[(long)n * NG + g] = 0.004f + (rand() % 50) * 1e-5f;
      hZp[(long)n * NG + g] = (float)(rand() % 16);
    }
  // Похідні суми: Sa' по (рядок, група) і Sq по (стовпець, група).
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < NG; ++g) {
      long s = 0;
      for (int w = 0; w < WORDS; ++w)
        for (int k = 0; k < 8; ++k) s += nib(hA[(long)m * K8 + g * WORDS + w], k);
      hSAp[(long)m * NG + g] = (float)s;
    }
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < NG; ++g) {
      long s = 0;
      for (int w = 0; w < WORDS; ++w)
        for (int k = 0; k < 8; ++k) s += nib(hB[(long)n * K8 + g * WORDS + w], k);
      hE[(long)n * NG + g] =
          8.f * hZp[(long)n * NG + g] * GS - 8.f * (float)s;
    }

  int *dA, *dB; float *dSAp, *dSw, *dZp, *dE, *dSa, *dC;
  hipMalloc(&dA, hA.size() * 4);     hipMalloc(&dB, hB.size() * 4);
  hipMalloc(&dSAp, hSAp.size() * 4); hipMalloc(&dSw, hSw.size() * 4);
  hipMalloc(&dZp, hZp.size() * 4);   hipMalloc(&dE, hE.size() * 4);
  hipMalloc(&dSa, hSa.size() * 4);   hipMalloc(&dC, (long)M * N * 4);
  hipMemcpy(dA, hA.data(), hA.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB.data(), hB.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dSAp, hSAp.data(), hSAp.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dSw, hSw.data(), hSw.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dZp, hZp.data(), hZp.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dE, hE.data(), hE.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dSa, hSa.data(), hSa.size() * 4, hipMemcpyHostToDevice);

  dim3 blk(128 * 64 / 32), grd((N + 63) / 64, (M + 127) / 128);
  for (int i = 0; i < 5; ++i)
    gemm_prodg1<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dSAp, dSw, dZp, dE, dSa, dC, M, N, K);
  hipDeviceSynchronize();
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0);
  for (int r = 0; r < 20; ++r)
    gemm_prodg1<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dSAp, dSw, dZp, dE, dSa, dC, M, N, K);
  hipEventRecord(e1); hipEventSynchronize(e1);
  float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 20;
  printf("  групові масштаби + одинарний sA: %7.3f мс  %6.2f ТОпс\n", ms,
         2.0 * M * N * K / (ms / 1e3) / 1e12);

  std::vector<float> hC((long)M * N);
  hipMemcpy(hC.data(), dC, (long)M * N * 4, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int t = 0; t < 12; ++t) {
    int m = rand() % M, n = rand() % N;
    double ref = 0;
    for (int g = 0; g < NG; ++g) {
      double s = 0;
      for (int w = 0; w < WORDS; ++w)
        for (int k = 0; k < 8; ++k) {
          double a = nib(hA[(long)m * K8 + g * WORDS + w], k) - 8.0;
          double q = nib(hB[(long)n * K8 + g * WORDS + w], k);
          s += a * (q - hZp[(long)n * NG + g]);
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
