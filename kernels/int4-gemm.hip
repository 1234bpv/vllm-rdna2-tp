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

template <int BM, int BN, int TM, int TN, int GPT>
__global__ __launch_bounds__(256) void gemm_i4(
    const int* __restrict__ A,    // [M, K/8]
    const int* __restrict__ B,    // [N, K/8]
    const float* __restrict__ As, // [M, K/GS]
    const float* __restrict__ Bs, // [N, K/GS]
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  // ⚠️ Мікроплитка ЧЕРЕЗКРОКОВА, а не суцільна. При суцільній сусідні смуги
  // читають рядки з кроком TN, і банки LDS повторюються через вісім —
  // дворазовий конфлікт лишається навіть із доповненням. Тут сусідні смуги
  // беруть СУСІДНІ рядки, крок між банками стає непарним (доповнений рядок),
  // і 32 смуги лягають у 32 банки.
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN;
  const int tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  // ⚠️ +1 слово доповнення: без нього рядок має рівно 16 слів, сусідні рядки
  // зсунуті на 16 банків із 32, і 32 смуги б'ються у два банки (16-кратний
  // конфлікт). Доповнення робить зсув непарним, тож рядки лягають у різні
  // банки.
  // GPT груп за одне завантаження: бар'єрів стає у GPT разів менше, а
  // внутрішній цикл довшає. Масштаби вимагають зведення в МЕЖАХ групи, тож
  // накопичувач скидається після кожної групи, а не в кінці плитки.
  __shared__ int sA[BM][GPT * WORDS + 1];
  __shared__ int sB[BN][GPT * WORDS + 1];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  for (int gt = 0; gt < NG; gt += GPT) {
    const int WT = GPT * WORDS;
    for (int t = tid; t < BM * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int m = m0 + r;
      sA[r][w] = (m < M) ? A[(long)m * K8 + gt * WORDS + w] : 0;
    }
    for (int t = tid; t < BN * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int n = n0 + r;
      sB[r][w] = (n < N) ? B[(long)n * K8 + gt * WORDS + w] : 0;
    }
    __syncthreads();

#pragma unroll 1
   for (int gg = 0; gg < GPT; ++gg) {
    const int g = gt + gg;

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0;

#pragma unroll 1
    for (int w = gg * WORDS; w < (gg + 1) * WORDS; ++w) {
      int av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = sA[tm + i * STM][w];
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = sB[tn + j * STN][w];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
      int m = m0 + tm + i * STM;
      float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int n = n0 + tn + j * STN;
        float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
        fout[i][j] += (float)acc[i][j] * as * bs;
      }
    }
   }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}


// ── Варіант 2: розкладка LDS [слово][рядок] і суцільна мікроплитка ──
// Задум: у варіанті 1 на 16 множень припадає 8 окремих читань з LDS плюс
// адресна арифметика — корисних інструкцій близько половини. Тут чотири
// потрібні рядки лежать ПОРУЧ, тож беруться одним 128-бітним завантаженням:
// 2 інструкції LDS замість 8 на ті самі 16 множень.
// Конфліктів банків це не додає: сусідні смуги читають сусідні четвірки слів,
// тобто суцільну ділянку, а не з кроком.
template <int BM, int BN, int TM, int TN, int GPT>
__global__ __launch_bounds__(256) void gemm_i4_v2(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STN = BN / TN;
  const int tm = (tid / STN) * TM;
  const int tn = (tid % STN) * TN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int WT = GPT * WORDS;

  __shared__ int sA[GPT * WORDS][BM];
  __shared__ int sB[GPT * WORDS][BN];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  for (int gt = 0; gt < NG; gt += GPT) {
    for (int t = tid; t < BM * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int m = m0 + r;
      sA[w][r] = (m < M) ? A[(long)m * K8 + gt * WORDS + w] : 0;
    }
    for (int t = tid; t < BN * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int n = n0 + r;
      sB[w][r] = (n < N) ? B[(long)n * K8 + gt * WORDS + w] : 0;
    }
    __syncthreads();

#pragma unroll 1
    for (int gg = 0; gg < GPT; ++gg) {
      const int g = gt + gg;
      int acc[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

#pragma unroll 1
      for (int w = gg * WORDS; w < (gg + 1) * WORDS; ++w) {
        int av[TM], bv[TN];
        // Суцільні четвірки — компілятор складає їх у ds_read_b128.
#pragma unroll
        for (int i = 0; i < TM; ++i) av[i] = sA[w][tm + i];
#pragma unroll
        for (int j = 0; j < TN; ++j) bv[j] = sB[w][tn + j];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j)
            acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int m = m0 + tm + i;
        float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int n = n0 + tn + j;
          float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
          fout[i][j] += (float)acc[i][j] * as * bs;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i, n = n0 + tn + j;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}


// ── Варіант 3: [слово][рядок] + ЯВНІ 128-бітні читання ──
// Варіант 2 провалився не через задум: я писав скалярний цикл і сподівався,
// що компілятор сам складе його у ds_read_b128. Він цього не робить, бо не
// може довести вирівнювання, — і виходили чотири окремі читання з кроком,
// тобто конфлікт банків.
//
// З документа ISA: конфлікти оцінюються ФАЗАМИ, і для b128 у фазі 8 смуг.
// Вісім смуг по чотири слова — це рівно 32 банки, тобто доступ безконфліктний
// за побудовою. Тут читання явні (тип int4), а вирівнювання гарантоване:
// масив вирівняно на 16 байтів, BM і BN кратні 4, tm і tn кратні TM/TN.
typedef int int4v __attribute__((ext_vector_type(4)));

template <int BM, int BN, int TM, int TN, int GPT>
__global__ __launch_bounds__(256) void gemm_i4_v3(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  static_assert(TM == 4 && TN == 4, "варіант 3 розрахований на мікроплитку 4x4");
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STN = BN / TN;
  const int tm = (tid / STN) * TM;
  const int tn = (tid % STN) * TN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int WT = GPT * WORDS;

  __shared__ __align__(16) int sA[GPT * WORDS][BM];
  __shared__ __align__(16) int sB[GPT * WORDS][BN];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  for (int gt = 0; gt < NG; gt += GPT) {
    // ⚠️ Порядок обходу: найшвидше змінюється РЯДОК, а не слово. У розкладці
    // [слово][рядок] сусідні слова стоять через BM позицій, тобто через
    // BM mod 32 = 0 банків — усі смуги писали б в ОДИН банк (32-кратний
    // конфлікт). При обході по рядку запис суцільний.
    for (int t = tid; t < BM * WT; t += nthr) {
      int w = t / BM, r = t % BM;
      int m = m0 + r;
      sA[w][r] = (m < M) ? A[(long)m * K8 + gt * WORDS + w] : 0;
    }
    for (int t = tid; t < BN * WT; t += nthr) {
      int w = t / BN, r = t % BN;
      int n = n0 + r;
      sB[w][r] = (n < N) ? B[(long)n * K8 + gt * WORDS + w] : 0;
    }
    __syncthreads();

#pragma unroll 1
    for (int gg = 0; gg < GPT; ++gg) {
      const int g = gt + gg;
      int acc[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

#pragma unroll 1
      for (int w = gg * WORDS; w < (gg + 1) * WORDS; ++w) {
        const int4v av = *(const int4v*)(&sA[w][tm]);
        const int4v bv = *(const int4v*)(&sB[w][tn]);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j)
            acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int m = m0 + tm + i;
        float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int n = n0 + tn + j;
          float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
          fout[i][j] += (float)acc[i][j] * as * bs;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i, n = n0 + tn + j;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}


// ── Варіант 4: варіант 1 + випереджальне читання LDS ──
// Замір складу інструкцій показав, що в гарячому блоці варіанта 1 із 47
// інструкцій сім — це s_waitcnt: множення чекають на щойно замовлені читання.
// Тут операнди наступного слова замовляються ДО того, як почнеться рахунок на
// поточному, тож очікування перекривається корисною роботою.
// ⚠️ Це НЕ те саме, що програмний конвеєр Triton, який на RDNA2 шкодив: там
// подвоювалися ТАЙЛИ в LDS і різалася зайнятість, тут же йдеться про вісім
// регістрів.
template <int BM, int BN, int TM, int TN, int GPT>
__global__ __launch_bounds__(256) void gemm_i4_v4(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int WT = GPT * WORDS;

  __shared__ int sA[BM][GPT * WORDS + 1];
  __shared__ int sB[BN][GPT * WORDS + 1];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  for (int gt = 0; gt < NG; gt += GPT) {
    for (int t = tid; t < BM * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int m = m0 + r;
      sA[r][w] = (m < M) ? A[(long)m * K8 + gt * WORDS + w] : 0;
    }
    for (int t = tid; t < BN * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int n = n0 + r;
      sB[r][w] = (n < N) ? B[(long)n * K8 + gt * WORDS + w] : 0;
    }
    __syncthreads();

#pragma unroll 1
    for (int gg = 0; gg < GPT; ++gg) {
      const int g = gt + gg;
      int acc[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

      const int w0 = gg * WORDS, w1 = (gg + 1) * WORDS;
      int av[TM], bv[TN], an[TM], bn[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = sA[tm + i * STM][w0];
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = sB[tn + j * STN][w0];

#pragma unroll 1
      for (int w = w0; w < w1; ++w) {
        // Замовляємо наступне слово ДО рахунку на поточному.
        if (w + 1 < w1) {
#pragma unroll
          for (int i = 0; i < TM; ++i) an[i] = sA[tm + i * STM][w + 1];
#pragma unroll
          for (int j = 0; j < TN; ++j) bn[j] = sB[tn + j * STN][w + 1];
        }
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j)
            acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
#pragma unroll
        for (int i = 0; i < TM; ++i) av[i] = an[i];
#pragma unroll
        for (int j = 0; j < TN; ++j) bv[j] = bn[j];
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int m = m0 + tm + i * STM;
        float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int n = n0 + tn + j * STN;
          float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
          fout[i][j] += (float)acc[i][j] * as * bs;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}


// ── Варіант 5: розкладка варіанта 1, але читання ПАРАМИ слів (ds_read_b64) ──
// Задум випливає з правила фаз: для b64 у фазі 16 смуг. При доповненні до
// парного кроку (WORDS+2 = 18 слів) шістнадцять смуг, що читають сусідні
// РЯДКИ, дають адреси 18*l + w; за модулем 32 це шістнадцять різних ПАРНИХ
// банків, а кожна смуга бере два сусідні слова — разом рівно 32 банки, тобто
// конфлікту немає. Інструкцій LDS при цьому вдвічі менше, ніж у варіанті 1.
typedef int int2v __attribute__((ext_vector_type(2)));

template <int BM, int BN, int TM, int TN, int GPT>
__global__ __launch_bounds__(256) void gemm_i4_v5(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int WT = GPT * WORDS;
  const int LD = GPT * WORDS + 2;  // ПАРНИЙ крок — потрібен для b64

  __shared__ __align__(8) int sA[BM][GPT * WORDS + 2];
  __shared__ __align__(8) int sB[BN][GPT * WORDS + 2];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  for (int gt = 0; gt < NG; gt += GPT) {
    for (int t = tid; t < BM * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int m = m0 + r;
      sA[r][w] = (m < M) ? A[(long)m * K8 + gt * WORDS + w] : 0;
    }
    for (int t = tid; t < BN * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int n = n0 + r;
      sB[r][w] = (n < N) ? B[(long)n * K8 + gt * WORDS + w] : 0;
    }
    __syncthreads();

#pragma unroll 1
    for (int gg = 0; gg < GPT; ++gg) {
      const int g = gt + gg;
      int acc[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

      // WORDS парне (16), а gg*WORDS теж парне, тож пари не перетинають межі
      // групи й вирівнювання на 8 байтів тримається.
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
        float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int n = n0 + tn + j * STN;
          float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
          fout[i][j] += (float)acc[i][j] * as * bs;
        }
      }
    }
    __syncthreads();
  }
  (void)LD;

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}


// ── Діагностичне ядро: та сама арифметика БЕЗ звернень до LDS у циклі ──
// ⚠️ РАХУЄ НЕПРАВИЛЬНО і призначене лише для заміру. Операнди беруться один
// раз до циклу, тож усередині лишаються тільки множення й лічильник. Якщо
// таке ядро дійде до стелі за щільністю — вузьким місцем є LDS; якщо
// впреться в ті самі ~29 ТОпс — межа деінде.
template <int BM, int BN, int TM, int TN, int GPT>
__global__ __launch_bounds__(256) void gemm_i4_diag(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  int av[TM], bv[TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) av[i] = A[(long)(m0 + tm + i * STM) % M * K8];
#pragma unroll
  for (int j = 0; j < TN; ++j) bv[j] = B[(long)(n0 + tn + j * STN) % N * K8];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  for (int g = 0; g < NG; ++g) {
    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0;
#pragma unroll 1
    for (int w = 0; w < WORDS; ++w) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_sdot8(av[i] + w, bv[j], acc[i][j], false);
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) fout[i][j] += (float)acc[i][j] * 1e-6f;
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}


// ── Варіант 6: обмін операндами між смугами через DPP (quad_perm) ──
// Замір показав, що ядро тримає кросбар LDS: без звернень до нього те саме
// ядро дає 63,9 ТОпс проти 29,1. Кросбар віддає 32 слова за такт незалежно
// від ширини інструкції, тож зменшити треба саме КІЛЬКІСТЬ прочитаних слів.
//
// Задум: смуги однієї четвірки потребують тих самих чотирьох стовпців B.
// Хай кожна читає лише ОДНЕ слово B, а решту три бере в сусідів через
// quad_perm — це модифікатор арифметичної інструкції, кросбар він не займає.
// Читань стає 4 (A) + 1 (B) = 5 замість 8.
//
// Розкладка: q = смуга в четвірці, Q = номер четвірки.
//   стовпці четвірки: cbase = (Q % 16) * 4, смуга читає cbase + q;
//   рядки смуги:      rbase = (Q / 16) * 16 + q * 4, далі rbase + i.
// Через це чотири смуги четвірки мають РІЗНІ рядки й ті самі стовпці, тобто
// не дублюють роботу. Читання A широкомовні (лише 4 різні адреси на хвилю),
// читання B при непарному кроці 17 дають 32 різні банки, тобто без конфліктів.
template <int BM, int BN, int GPT>
__global__ __launch_bounds__(256) void gemm_i4_v6(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int TM = 4, TN = 4;
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int q = tid & 3, Q = tid >> 2;
  const int cbase = (Q % (BN / 4)) * 4;
  const int rbase = (Q / (BN / 4)) * (4 * TM) + q * TM;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int WT = GPT * WORDS;

  __shared__ int sA[BM][GPT * WORDS + 1];
  __shared__ int sB[BN][GPT * WORDS + 1];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  for (int gt = 0; gt < NG; gt += GPT) {
    for (int t = tid; t < BM * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int m = m0 + r;
      sA[r][w] = (m < M) ? A[(long)m * K8 + gt * WORDS + w] : 0;
    }
    for (int t = tid; t < BN * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int n = n0 + r;
      sB[r][w] = (n < N) ? B[(long)n * K8 + gt * WORDS + w] : 0;
    }
    __syncthreads();

#pragma unroll 1
    for (int gg = 0; gg < GPT; ++gg) {
      const int g = gt + gg;
      int acc[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

#pragma unroll 1
      for (int w = gg * WORDS; w < (gg + 1) * WORDS; ++w) {
        int av[TM];
#pragma unroll
        for (int i = 0; i < TM; ++i) av[i] = sA[rbase + i][w];
        const int bmine = sB[cbase + q][w];
        // quad_perm[k,k,k,k]: усі чотири смуги четвірки беруть значення
        // смуги k. Коди 0x00, 0x55, 0xAA, 0xFF.
        int bv[TN];
        bv[0] = __builtin_amdgcn_mov_dpp(bmine, 0x00, 0xf, 0xf, true);
        bv[1] = __builtin_amdgcn_mov_dpp(bmine, 0x55, 0xf, 0xf, true);
        bv[2] = __builtin_amdgcn_mov_dpp(bmine, 0xAA, 0xf, 0xf, true);
        bv[3] = __builtin_amdgcn_mov_dpp(bmine, 0xFF, 0xf, 0xf, true);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j)
            acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int m = m0 + rbase + i;
        float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int n = n0 + cbase + j;
          float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
          fout[i][j] += (float)acc[i][j] * as * bs;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + rbase + i, n = n0 + cbase + j;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}


// ── Діагностика 2: стягування ЛИШАЄТЬСЯ, зникають лише читання LDS у циклі ──
// ⚠️ Перша діагностика була хибна: вона прибирала й читання з LDS, І все
// стягування з глобальної пам'яті, тож міряла суму двох причин. Тут глобальні
// читання, записи в LDS і бар'єри лишаються точно як у робочому ядрі, а в
// внутрішньому циклі операнди беруться з регістрів. РАХУЄ НЕПРАВИЛЬНО.
template <int BM, int BN, int TM, int TN, int GPT>
__global__ __launch_bounds__(256) void gemm_i4_diag2(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int WT = GPT * WORDS;

  __shared__ int sA[BM][GPT * WORDS + 1];
  __shared__ int sB[BN][GPT * WORDS + 1];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  int av[TM], bv[TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) av[i] = i + tid;
#pragma unroll
  for (int j = 0; j < TN; ++j) bv[j] = j - tid;

  for (int gt = 0; gt < NG; gt += GPT) {
    for (int t = tid; t < BM * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int m = m0 + r;
      sA[r][w] = (m < M) ? A[(long)m * K8 + gt * WORDS + w] : 0;
    }
    for (int t = tid; t < BN * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int n = n0 + r;
      sB[r][w] = (n < N) ? B[(long)n * K8 + gt * WORDS + w] : 0;
    }
    __syncthreads();

#pragma unroll 1
    for (int gg = 0; gg < GPT; ++gg) {
      const int g = gt + gg;
      int acc[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;
#pragma unroll 1
      for (int w = gg * WORDS; w < (gg + 1) * WORDS; ++w) {
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j)
            acc[i][j] = __builtin_amdgcn_sdot8(av[i] + w, bv[j], acc[i][j], false);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int m = m0 + tm + i * STM;
        float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          int n = n0 + tn + j * STN;
          float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
          fout[i][j] += (float)acc[i][j] * as * bs;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}


// ── Варіант 7: та сама схема, але ПЛИТКА 128x128 і 1024 нитки ──
// Діагностика 2 показала, що ядро тримає стягування плиток із глобальної
// пам'яті, а не кросбар LDS: прибирання ВСІХ читань LDS у циклі дало лише
// +5%. Отже важіль — менше повторних читань, тобто більша плитка. Раніше
// 128x128 програвала, бо йшла з мікроплиткою 8x8 і 249 регістрами; тут
// мікроплитка лишається 4x4, а нитки додаються — регістрів це не чіпає.
template <int BM, int BN, int TM, int TN, int GPT>
__global__ __launch_bounds__(1024) void gemm_i4_v7(
    const int* __restrict__ A,    // [M, K/8]
    const int* __restrict__ B,    // [N, K/8]
    const float* __restrict__ As, // [M, K/GS]
    const float* __restrict__ Bs, // [N, K/GS]
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  // ⚠️ Мікроплитка ЧЕРЕЗКРОКОВА, а не суцільна. При суцільній сусідні смуги
  // читають рядки з кроком TN, і банки LDS повторюються через вісім —
  // дворазовий конфлікт лишається навіть із доповненням. Тут сусідні смуги
  // беруть СУСІДНІ рядки, крок між банками стає непарним (доповнений рядок),
  // і 32 смуги лягають у 32 банки.
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN;
  const int tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  // ⚠️ +1 слово доповнення: без нього рядок має рівно 16 слів, сусідні рядки
  // зсунуті на 16 банків із 32, і 32 смуги б'ються у два банки (16-кратний
  // конфлікт). Доповнення робить зсув непарним, тож рядки лягають у різні
  // банки.
  // GPT груп за одне завантаження: бар'єрів стає у GPT разів менше, а
  // внутрішній цикл довшає. Масштаби вимагають зведення в МЕЖАХ групи, тож
  // накопичувач скидається після кожної групи, а не в кінці плитки.
  __shared__ int sA[BM][GPT * WORDS + 1];
  __shared__ int sB[BN][GPT * WORDS + 1];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  for (int gt = 0; gt < NG; gt += GPT) {
    const int WT = GPT * WORDS;
    for (int t = tid; t < BM * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int m = m0 + r;
      sA[r][w] = (m < M) ? A[(long)m * K8 + gt * WORDS + w] : 0;
    }
    for (int t = tid; t < BN * WT; t += nthr) {
      int r = t / WT, w = t % WT;
      int n = n0 + r;
      sB[r][w] = (n < N) ? B[(long)n * K8 + gt * WORDS + w] : 0;
    }
    __syncthreads();

#pragma unroll 1
   for (int gg = 0; gg < GPT; ++gg) {
    const int g = gt + gg;

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0;

#pragma unroll 1
    for (int w = gg * WORDS; w < (gg + 1) * WORDS; ++w) {
      int av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = sA[tm + i * STM][w];
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = sB[tn + j * STN][w];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
      int m = m0 + tm + i * STM;
      float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int n = n0 + tn + j * STN;
        float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
        fout[i][j] += (float)acc[i][j] * as * bs;
      }
    }
   }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int m = m0 + tm + i * STM, n = n0 + tn + j * STN;
      if (m < M && n < N) C[(long)m * N + n] = fout[i][j];
    }
}



// ── Варіант 8: подвійна буферизація LDS + випереджальне читання з глобальної ──
// Діагностика 2 (стягування є, читань LDS у циклі немає) дала 30,60 проти
// 29,06 у робочого ядра, тобто внутрішній цикл майже безкоштовний. Більша
// плитка, яка вдвічі зменшує ОБСЯГ глобального трафіку, теж не допомогла
// (24,7). Отже коштує не обсяг, а ОЧІКУВАННЯ: після бар'єра всі нитки стоять,
// поки прийде наступна плитка.
//
// Тут глобальні читання наступної плитки замовляються ДО рахунку на поточній,
// а плитки лежать у двох буферах, тож бар'єр на групу лишається один.
// ⚠️ Це не те подвоєння тайлів, що шкодило в Triton-ядрах уваги: там воно
// різало зайнятість удвічі. Тут LDS 34,8 КіБ із 64, а регістрів додається
// вісім.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_v8(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int PA = BM * WORDS / nthr;   // скільки слів A на нитку
  const int PB = BN * WORDS / nthr;   // скільки слів B на нитку

  __shared__ int sA[2][BM][WORDS + 1];
  __shared__ int sB[2][BN][WORDS + 1];

  float fout[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) fout[i][j] = 0.f;

  int ra[PA > 0 ? PA : 1], rb[PB > 0 ? PB : 1];

  // Перша плитка.
#pragma unroll
  for (int k = 0; k < PA; ++k) {
    int t = tid + k * nthr, r = t / WORDS, w = t % WORDS;
    int m = m0 + r;
    sA[0][r][w] = (m < M) ? A[(long)m * K8 + w] : 0;
  }
#pragma unroll
  for (int k = 0; k < PB; ++k) {
    int t = tid + k * nthr, r = t / WORDS, w = t % WORDS;
    int n = n0 + r;
    sB[0][r][w] = (n < N) ? B[(long)n * K8 + w] : 0;
  }
  __syncthreads();

  for (int g = 0; g < NG; ++g) {
    const int cur = g & 1, nxt = cur ^ 1;
    // Замовляємо наступну плитку ДО рахунку — читання летять паралельно.
    if (g + 1 < NG) {
#pragma unroll
      for (int k = 0; k < PA; ++k) {
        int t = tid + k * nthr, r = t / WORDS, w = t % WORDS;
        int m = m0 + r;
        ra[k] = (m < M) ? A[(long)m * K8 + (g + 1) * WORDS + w] : 0;
      }
#pragma unroll
      for (int k = 0; k < PB; ++k) {
        int t = tid + k * nthr, r = t / WORDS, w = t % WORDS;
        int n = n0 + r;
        rb[k] = (n < N) ? B[(long)n * K8 + (g + 1) * WORDS + w] : 0;
      }
    }

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0;
#pragma unroll 1
    for (int w = 0; w < WORDS; ++w) {
      int av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = sA[cur][tm + i * STM][w];
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = sB[cur][tn + j * STN][w];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      int m = m0 + tm + i * STM;
      float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int n = n0 + tn + j * STN;
        float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
        fout[i][j] += (float)acc[i][j] * as * bs;
      }
    }

    if (g + 1 < NG) {
      __syncthreads();
#pragma unroll
      for (int k = 0; k < PA; ++k) {
        int t = tid + k * nthr, r = t / WORDS, w = t % WORDS;
        sA[nxt][r][w] = ra[k];
      }
#pragma unroll
      for (int k = 0; k < PB; ++k) {
        int t = tid + k * nthr, r = t / WORDS, w = t % WORDS;
        sB[nxt][r][w] = rb[k];
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


// ── Варіант 9: варіант 8 + ШИРОКІ глобальні читання ──
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_v9(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int PA = BM * WORDS / nthr;   // скільки слів A на нитку
  const int PB = BN * WORDS / nthr;   // скільки слів B на нитку

  __shared__ int sA[2][BM][WORDS + 1];
  __shared__ int sB[2][BN][WORDS + 1];

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
  const int W4 = WORDS / 4;          // 128-бітних порцій у рядку
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
  __syncthreads();

  for (int g = 0; g < NG; ++g) {
    const int cur = g & 1, nxt = cur ^ 1;
    // Замовляємо наступну плитку ДО рахунку — читання летять паралельно.
    if (g + 1 < NG) {
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int m = m0 + r;
        ra[k] = (m < M) ? *(const int4v*)(&A[(long)m * K8 + (g + 1) * WORDS + w])
                        : (int4v)(0);
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int n = n0 + r;
        rb[k] = (n < N) ? *(const int4v*)(&B[(long)n * K8 + (g + 1) * WORDS + w])
                        : (int4v)(0);
      }
    }

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0;
#pragma unroll 1
    for (int w = 0; w < WORDS; ++w) {
      int av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = sA[cur][tm + i * STM][w];
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = sB[cur][tn + j * STN][w];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      int m = m0 + tm + i * STM;
      float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int n = n0 + tn + j * STN;
        float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
        fout[i][j] += (float)acc[i][j] * as * bs;
      }
    }

    if (g + 1 < NG) {
      __syncthreads();
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

// ── Варіант 10: варіант 9 з ОДНИМ бар'єром на групу ──
// При подвійному буфері другий бар'єр зайвий: у групі g нитки читають буфер
// cur і пишуть у nxt, тож ніхто не читає того, у що пишуть. Єдине, що треба
// гарантувати, — що читання буфера nxt на кроці g-1 завершилися; це й робить
// бар'єр у кінці кроку g-1.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_v10(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int PA = BM * WORDS / nthr;   // скільки слів A на нитку
  const int PB = BN * WORDS / nthr;   // скільки слів B на нитку

  __shared__ int sA[2][BM][WORDS + 1];
  __shared__ int sB[2][BN][WORDS + 1];

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
  const int W4 = WORDS / 4;          // 128-бітних порцій у рядку
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
  __syncthreads();

  for (int g = 0; g < NG; ++g) {
    const int cur = g & 1, nxt = cur ^ 1;
    // Замовляємо наступну плитку ДО рахунку — читання летять паралельно.
    if (g + 1 < NG) {
#pragma unroll
      for (int k = 0; k < PA / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int m = m0 + r;
        ra[k] = (m < M) ? *(const int4v*)(&A[(long)m * K8 + (g + 1) * WORDS + w])
                        : (int4v)(0);
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int n = n0 + r;
        rb[k] = (n < N) ? *(const int4v*)(&B[(long)n * K8 + (g + 1) * WORDS + w])
                        : (int4v)(0);
      }
    }

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0;
#pragma unroll 1
    for (int w = 0; w < WORDS; ++w) {
      int av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = sA[cur][tm + i * STM][w];
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = sB[cur][tn + j * STN][w];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      int m = m0 + tm + i * STM;
      float as = (m < M) ? As[(long)m * NG + g] : 0.f;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int n = n0 + tn + j * STN;
        float bs = (n < N) ? Bs[(long)n * NG + g] : 0.f;
        fout[i][j] += (float)acc[i][j] * as * bs;
      }
    }

    if (g + 1 < NG) {
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

// ── Варіант 11: варіант 10, але ДВІ групи на одне стягування ──
// Бар'єрів стає вдвічі менше, а роботи між ними вдвічі більше. Ціна — удвічі
// більший буфер LDS і вдвічі більше регістрів під випереджальне читання.
// При подвійному буфері другий бар'єр зайвий: у групі g нитки читають буфер
// cur і пишуть у nxt, тож ніхто не читає того, у що пишуть. Єдине, що треба
// гарантувати, — що читання буфера nxt на кроці g-1 завершилися; це й робить
// бар'єр у кінці кроку g-1.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_v11(
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

  __shared__ int sA[2][BM][2 * WORDS + 1];
  __shared__ int sB[2][BN][2 * WORDS + 1];

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
    for (int w = gg * WORDS; w < (gg + 1) * WORDS; ++w) {
      int av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = sA[cur][tm + i * STM][w];
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = sB[cur][tn + j * STN][w];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      int m = m0 + tm + i * STM;
      float as = (m < M) ? As[(long)m * NG + g + gg] : 0.f;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int n = n0 + tn + j * STN;
        float bs = (n < N) ? Bs[(long)n * NG + g + gg] : 0.f;
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

// ── Варіант 12: масштаби груп теж через LDS ──
// Ізоляція чинників показала, що головний пожирач — обробка масштабів:
// без них те саме ядро дає 73,2 ТОпс замість 36,7. Причина: As і Bs читалися
// з ГЛОБАЛЬНОЇ пам'яті просто в епілозі кожної групи — 8+4 читання на нитку
// з повною затримкою, і від них залежить перетворення накопичувачів.
// Тут вони стягуються в LDS разом із плитками: 128+64 значень на групу проти
// 256x12 окремих читань.
// Бар'єрів стає вдвічі менше, а роботи між ними вдвічі більше. Ціна — удвічі
// більший буфер LDS і вдвічі більше регістрів під випереджальне читання.
// При подвійному буфері другий бар'єр зайвий: у групі g нитки читають буфер
// cur і пишуть у nxt, тож ніхто не читає того, у що пишуть. Єдине, що треба
// гарантувати, — що читання буфера nxt на кроці g-1 завершилися; це й робить
// бар'єр у кінці кроку g-1.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_v12(
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

  __shared__ int sA[2][BM][2 * WORDS + 1];
  __shared__ int sB[2][BN][2 * WORDS + 1];
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
    for (int w = gg * WORDS; w < (gg + 1) * WORDS; ++w) {
      int av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = sA[cur][tm + i * STM][w];
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = sB[cur][tn + j * STN][w];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
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

// ── Варіант 13: варіант 12 + читання LDS ПАРАМИ слів ──
// Розгортання циклу перевірено й відкинуто (50,1 -> 46,2 при множнику 8):
// кожна розгорнута ітерація тримає ще один комплект адрес, регістри ростуть
// зі 136 до 220. Мети розгортання — більшої частки корисних інструкцій —
// досягнуто інакше: пара слів за одне ds_read_b64 дає 6 читань на слово
// замість 12.
// ⚠️ Той самий прийом у варіанті 5 не дав нічого: тоді межею були масштаби.
// Ізоляція чинників показала, що головний пожирач — обробка масштабів:
// без них те саме ядро дає 73,2 ТОпс замість 36,7. Причина: As і Bs читалися
// з ГЛОБАЛЬНОЇ пам'яті просто в епілозі кожної групи — 8+4 читання на нитку
// з повною затримкою, і від них залежить перетворення накопичувачів.
// Тут вони стягуються в LDS разом із плитками: 128+64 значень на групу проти
// 256x12 окремих читань.
// Бар'єрів стає вдвічі менше, а роботи між ними вдвічі більше. Ціна — удвічі
// більший буфер LDS і вдвічі більше регістрів під випереджальне читання.
// При подвійному буфері другий бар'єр зайвий: у групі g нитки читають буфер
// cur і пишуть у nxt, тож ніхто не читає того, у що пишуть. Єдине, що треба
// гарантувати, — що читання буфера nxt на кроці g-1 завершилися; це й робить
// бар'єр у кінці кроку g-1.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_v13(
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

// ── Варіант 16: варіант 13 + s_setprio навколо підкачки ──
// Прийом із асемблерних ядер Tensile: перед видачею глобальних читань
// підняти пріоритет хвилі, після — повернути. Планувальник тоді видає
// підкачку наперед і не дає їй загубитися серед множень.
// Розгортання циклу перевірено й відкинуто (50,1 -> 46,2 при множнику 8):
// кожна розгорнута ітерація тримає ще один комплект адрес, регістри ростуть
// зі 136 до 220. Мети розгортання — більшої частки корисних інструкцій —
// досягнуто інакше: пара слів за одне ds_read_b64 дає 6 читань на слово
// замість 12.
// ⚠️ Той самий прийом у варіанті 5 не дав нічого: тоді межею були масштаби.
// Ізоляція чинників показала, що головний пожирач — обробка масштабів:
// без них те саме ядро дає 73,2 ТОпс замість 36,7. Причина: As і Bs читалися
// з ГЛОБАЛЬНОЇ пам'яті просто в епілозі кожної групи — 8+4 читання на нитку
// з повною затримкою, і від них залежить перетворення накопичувачів.
// Тут вони стягуються в LDS разом із плитками: 128+64 значень на групу проти
// 256x12 окремих читань.
// Бар'єрів стає вдвічі менше, а роботи між ними вдвічі більше. Ціна — удвічі
// більший буфер LDS і вдвічі більше регістрів під випереджальне читання.
// При подвійному буфері другий бар'єр зайвий: у групі g нитки читають буфер
// cur і пишуть у nxt, тож ніхто не читає того, у що пишуть. Єдине, що треба
// гарантувати, — що читання буфера nxt на кроці g-1 завершилися; це й робить
// бар'єр у кінці кроку g-1.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_v16(
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
      __builtin_amdgcn_s_setprio(1);
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
    __builtin_amdgcn_s_setprio(0);
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

// ── Варіант 17: варіант 13 + перестановка порядку блоків ──
// Замість лінійного обходу (усі стовпці одного рядка підряд) блоки групуються
// смугами по GROUP рядків. Сусідні блоки тоді ділять ті самі плитки B, і ті
// довше лишаються в кеші.
// Розгортання циклу перевірено й відкинуто (50,1 -> 46,2 при множнику 8):
// кожна розгорнута ітерація тримає ще один комплект адрес, регістри ростуть
// зі 136 до 220. Мети розгортання — більшої частки корисних інструкцій —
// досягнуто інакше: пара слів за одне ds_read_b64 дає 6 читань на слово
// замість 12.
// ⚠️ Той самий прийом у варіанті 5 не дав нічого: тоді межею були масштаби.
// Ізоляція чинників показала, що головний пожирач — обробка масштабів:
// без них те саме ядро дає 73,2 ТОпс замість 36,7. Причина: As і Bs читалися
// з ГЛОБАЛЬНОЇ пам'яті просто в епілозі кожної групи — 8+4 читання на нитку
// з повною затримкою, і від них залежить перетворення накопичувачів.
// Тут вони стягуються в LDS разом із плитками: 128+64 значень на групу проти
// 256x12 окремих читань.
// Бар'єрів стає вдвічі менше, а роботи між ними вдвічі більше. Ціна — удвічі
// більший буфер LDS і вдвічі більше регістрів під випереджальне читання.
// При подвійному буфері другий бар'єр зайвий: у групі g нитки читають буфер
// cur і пишуть у nxt, тож ніхто не читає того, у що пишуть. Єдине, що треба
// гарантувати, — що читання буфера nxt на кроці g-1 завершилися; це й робить
// бар'єр у кінці кроку g-1.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_v17(
    const int* __restrict__ A, const int* __restrict__ B,
    const float* __restrict__ As, const float* __restrict__ Bs,
    float* __restrict__ C, int M, int N, int K) {
  const int K8 = K / 8, NG = K / GS;
  const int tid = threadIdx.x;
  const int nthr = BM * BN / (TM * TN);
  const int STM = BM / TM, STN = BN / TN;
  const int tm = tid / STN, tn = tid % STN;
  // Групування блоків смугами по GROUP рядків.
  const int GROUP = 8;
  const int nb_n = gridDim.x, nb_m = gridDim.y;
  const int lin = blockIdx.y * nb_n + blockIdx.x;
  const int per_group = GROUP * nb_n;
  const int grp = lin / per_group;
  const int in_grp = lin % per_group;
  const int rows_here = (nb_m - grp * GROUP) < GROUP ? (nb_m - grp * GROUP) : GROUP;
  const int bm_i = grp * GROUP + in_grp % rows_here;
  const int bn_i = in_grp / rows_here;
  if (bm_i >= nb_m || bn_i >= nb_n) return;
  const int m0 = bm_i * BM, n0 = bn_i * BN;
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

// ── Діагностика 4: варіант 13 БЕЗ читань LDS у циклі ──
// ⚠️ РАХУЄ НЕПРАВИЛЬНО. Стягування, подвійний буфер, бар'єри й уся арифметика
// масштабів лишаються на місці; операнди множення беруться з регістрів,
// набраних до циклу. Показує ціну самих читань з LDS.
// Розгортання циклу перевірено й відкинуто (50,1 -> 46,2 при множнику 8):
// кожна розгорнута ітерація тримає ще один комплект адрес, регістри ростуть
// зі 136 до 220. Мети розгортання — більшої частки корисних інструкцій —
// досягнуто інакше: пара слів за одне ds_read_b64 дає 6 читань на слово
// замість 12.
// ⚠️ Той самий прийом у варіанті 5 не дав нічого: тоді межею були масштаби.
// Ізоляція чинників показала, що головний пожирач — обробка масштабів:
// без них те саме ядро дає 73,2 ТОпс замість 36,7. Причина: As і Bs читалися
// з ГЛОБАЛЬНОЇ пам'яті просто в епілозі кожної групи — 8+4 читання на нитку
// з повною затримкою, і від них залежить перетворення накопичувачів.
// Тут вони стягуються в LDS разом із плитками: 128+64 значень на групу проти
// 256x12 окремих читань.
// Бар'єрів стає вдвічі менше, а роботи між ними вдвічі більше. Ціна — удвічі
// більший буфер LDS і вдвічі більше регістрів під випереджальне читання.
// При подвійному буфері другий бар'єр зайвий: у групі g нитки читають буфер
// cur і пишуть у nxt, тож ніхто не читає того, у що пишуть. Єдине, що треба
// гарантувати, — що читання буфера nxt на кроці g-1 завершилися; це й робить
// бар'єр у кінці кроку g-1.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_diag4(
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
      for (int i = 0; i < TM; ++i) { av[i][0] = tm + i + w; av[i][1] = tm - i; }
#pragma unroll
      for (int j = 0; j < TN; ++j) { bv[j][0] = tn + j; bv[j][1] = tn - j + w; }
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

// ── Діагностика 5: ні читань LDS, ні стягування ──
// ⚠️ РАХУЄ НЕПРАВИЛЬНО. Лишаються тільки множення, епілог масштабів і
// керування циклом. Різниця з Д4 показує ціну стягування плиток.
// ⚠️ РАХУЄ НЕПРАВИЛЬНО. Стягування, подвійний буфер, бар'єри й уся арифметика
// масштабів лишаються на місці; операнди множення беруться з регістрів,
// набраних до циклу. Показує ціну самих читань з LDS.
// Розгортання циклу перевірено й відкинуто (50,1 -> 46,2 при множнику 8):
// кожна розгорнута ітерація тримає ще один комплект адрес, регістри ростуть
// зі 136 до 220. Мети розгортання — більшої частки корисних інструкцій —
// досягнуто інакше: пара слів за одне ds_read_b64 дає 6 читань на слово
// замість 12.
// ⚠️ Той самий прийом у варіанті 5 не дав нічого: тоді межею були масштаби.
// Ізоляція чинників показала, що головний пожирач — обробка масштабів:
// без них те саме ядро дає 73,2 ТОпс замість 36,7. Причина: As і Bs читалися
// з ГЛОБАЛЬНОЇ пам'яті просто в епілозі кожної групи — 8+4 читання на нитку
// з повною затримкою, і від них залежить перетворення накопичувачів.
// Тут вони стягуються в LDS разом із плитками: 128+64 значень на групу проти
// 256x12 окремих читань.
// Бар'єрів стає вдвічі менше, а роботи між ними вдвічі більше. Ціна — удвічі
// більший буфер LDS і вдвічі більше регістрів під випереджальне читання.
// При подвійному буфері другий бар'єр зайвий: у групі g нитки читають буфер
// cur і пишуть у nxt, тож ніхто не читає того, у що пишуть. Єдине, що треба
// гарантувати, — що читання буфера nxt на кроці g-1 завершилися; це й робить
// бар'єр у кінці кроку g-1.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_diag5(
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
    int4v v = (int4v)(w + m);
#pragma unroll
    for (int e = 0; e < 4; ++e) sA[0][r][w + e] = v[e];
  }
#pragma unroll
  for (int k = 0; k < PB / 4; ++k) {
    int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
    int n = n0 + r;
    int4v v = (int4v)(w - n);
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
        ra[k] = (int4v)(w + m);
      }
#pragma unroll
      for (int k = 0; k < PB / 4; ++k) {
        int r = tid / W4 + k * rowsA, w = (tid % W4) * 4;
        int n = n0 + r;
        rb[k] = (int4v)(w - n);
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
      for (int i = 0; i < TM; ++i) { av[i][0] = tm + i + w; av[i][1] = tm - i; }
#pragma unroll
      for (int j = 0; j < TN; ++j) { bv[j][0] = tn + j; bv[j][1] = tn - j + w; }
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

// ── Діагностика 3: варіант 13 БЕЗ арифметики масштабів ──
// ⚠️ РАХУЄ НЕПРАВИЛЬНО. Показує ціну епілогу: три інструкції на кожен із 32
// накопичувачів (перетворення у float і два множення), і так щогрупи.
// Ізоляція чинників показала, що головний пожирач — обробка масштабів:
// без них те саме ядро дає 73,2 ТОпс замість 36,7. Причина: As і Bs читалися
// з ГЛОБАЛЬНОЇ пам'яті просто в епілозі кожної групи — 8+4 читання на нитку
// з повною затримкою, і від них залежить перетворення накопичувачів.
// Тут вони стягуються в LDS разом із плитками: 128+64 значень на групу проти
// 256x12 окремих читань.
// Бар'єрів стає вдвічі менше, а роботи між ними вдвічі більше. Ціна — удвічі
// більший буфер LDS і вдвічі більше регістрів під випереджальне читання.
// При подвійному буфері другий бар'єр зайвий: у групі g нитки читають буфер
// cur і пишуть у nxt, тож ніхто не читає того, у що пишуть. Єдине, що треба
// гарантувати, — що читання буфера nxt на кроці g-1 завершилися; це й робить
// бар'єр у кінці кроку g-1.
template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_i4_diag3(
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
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        fout[i][j] += (float)(acc[i][j] & 1);
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

static inline int nibble(int word, int k) {  // знаковий 4-бітний
  int v = (word >> (4 * k)) & 0xF;
  return (v & 8) ? v - 16 : v;
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

  double best = 1e30; const char* bestName = "";
#define PROBA(BM, BN, TM, TN, GPT)                                            \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  %3dx%-3d / %dx%d груп=%d : %7.3f мс  %6.2f ТОпс\n", BM, BN,   \
             TM, TN, GPT, ms, tops);                                          \
      if (ms < best) { best = ms; bestName = #BM "x" #BN "/" #TM "x" #TN "/g" #GPT; } \
    }                                                                         \
  }
#define PROBA2(BM, BN, TM, TN, GPT)                                           \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v2<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v2<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в2 %3dx%-3d / %dx%d груп=%d : %7.3f мс  %6.2f ТОпс\n", BM,    \
             BN, TM, TN, GPT, ms, tops);                                      \
      if (ms < best) { best = ms; bestName = "в2 " #BM "x" #BN "/" #TM "x" #TN "/g" #GPT; } \
    }                                                                         \
  }
  PROBA(64, 64, 4, 4, 1)
  PROBA(64, 64, 4, 4, 2)
  PROBA(64, 64, 4, 4, 4)
  PROBA(128, 64, 8, 4, 2)
  PROBA(128, 64, 8, 4, 4)
  PROBA(64, 64, 8, 8, 4)
  PROBA(128, 128, 8, 8, 2)
  PROBA2(64, 64, 4, 4, 1)
  PROBA2(64, 64, 4, 4, 2)
  PROBA2(128, 64, 8, 4, 2)
  PROBA2(128, 128, 8, 8, 2)
  PROBA2(64, 128, 4, 4, 2)
#define PROBA3(BM, BN, TM, TN, GPT)                                           \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v3<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v3<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в3 %3dx%-3d / %dx%d груп=%d : %7.3f мс  %6.2f ТОпс\n", BM,    \
             BN, TM, TN, GPT, ms, tops);                                      \
      if (ms < best) { best = ms; bestName = "в3 " #BM "x" #BN "/" #TM "x" #TN "/g" #GPT; } \
    }                                                                         \
  }
  PROBA3(64, 64, 4, 4, 1)
  PROBA3(64, 64, 4, 4, 2)
  PROBA3(64, 64, 4, 4, 4)
  PROBA3(128, 64, 4, 4, 2)
  PROBA3(64, 128, 4, 4, 2)
  PROBA3(128, 128, 4, 4, 2)
#define PROBA4(BM, BN, TM, TN, GPT)                                           \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v4<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v4<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в4 %3dx%-3d / %dx%d груп=%d : %7.3f мс  %6.2f ТОпс\n", BM,    \
             BN, TM, TN, GPT, ms, tops);                                      \
      if (ms < best) { best = ms; bestName = "в4 " #BM "x" #BN "/" #TM "x" #TN "/g" #GPT; } \
    }                                                                         \
  }
  PROBA4(64, 64, 4, 4, 1)
  PROBA4(64, 64, 4, 4, 2)
  PROBA4(128, 64, 8, 4, 2)
  PROBA4(64, 64, 4, 8, 2)
  PROBA4(64, 64, 8, 4, 2)
#define PROBA5(BM, BN, TM, TN, GPT)                                           \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v5<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v5<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в5 %3dx%-3d / %dx%d груп=%d : %7.3f мс  %6.2f ТОпс\n", BM,    \
             BN, TM, TN, GPT, ms, tops);                                      \
      if (ms < best) { best = ms; bestName = "в5 " #BM "x" #BN "/" #TM "x" #TN "/g" #GPT; } \
    }                                                                         \
  }
  PROBA5(64, 64, 4, 4, 1)
  PROBA5(64, 64, 4, 4, 2)
  PROBA5(128, 64, 8, 4, 2)
  PROBA5(64, 64, 4, 8, 2)
#define PROBA6(BM, BN, GPT)                                                   \
  {                                                                           \
    dim3 blk(BM * BN / 16);                                                   \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v6<BM, BN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);  \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v6<BM, BN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);  \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в6-DPP %3dx%-3d груп=%d : %7.3f мс  %6.2f ТОпс\n", BM, BN,    \
             GPT, ms, tops);                                                  \
      if (ms < best) { best = ms; bestName = "в6-DPP " #BM "x" #BN "/g" #GPT; }\
    }                                                                         \
  }
  PROBA6(64, 64, 1)
  PROBA6(64, 64, 2)
  PROBA6(64, 64, 4)
#define PROBA7(BM, BN, TM, TN, GPT)                                           \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 1024) {                                                      \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v7<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v7<BM, BN, TM, TN, GPT><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в7 %3dx%-3d / %dx%d ниток=%4d груп=%d : %7.3f мс  %6.2f ТОпс\n", \
             BM, BN, TM, TN, (int)blk.x, GPT, ms, tops);                      \
      if (ms < best) { best = ms; bestName = "в7 " #BM "x" #BN "/" #TM "x" #TN "/g" #GPT; } \
    }                                                                         \
  }
  PROBA7(128, 128, 4, 4, 1)
  PROBA7(128, 128, 4, 4, 2)
  PROBA7(128, 256, 4, 4, 1)
  PROBA7(256, 128, 4, 4, 1)
  PROBA7(128, 128, 8, 4, 1)
  PROBA7(128, 128, 4, 8, 1)
#define PROBA8(BM, BN, TM, TN)                                                \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v8<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v8<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в8-подв.буфер %3dx%-3d / %dx%d : %7.3f мс  %6.2f ТОпс\n",     \
             BM, BN, TM, TN, ms, tops);                                       \
      if (ms < best) { best = ms; bestName = "в8 " #BM "x" #BN "/" #TM "x" #TN; } \
    }                                                                         \
  }
  PROBA8(64, 64, 4, 4)
  PROBA8(128, 64, 8, 4)
  PROBA8(64, 128, 4, 8)
#define PROBA9(BM, BN, TM, TN)                                                \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v9<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v9<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в9-широкі %3dx%-3d / %dx%d : %7.3f мс  %6.2f ТОпс\n",         \
             BM, BN, TM, TN, ms, tops);                                       \
      if (ms < best) { best = ms; bestName = "в9 " #BM "x" #BN "/" #TM "x" #TN; } \
    }                                                                         \
  }
  PROBA9(64, 64, 4, 4)
  PROBA9(128, 64, 8, 4)
  PROBA9(128, 64, 4, 4)
  PROBA9(128, 128, 8, 8)
  PROBA9(128, 128, 4, 8)
  PROBA9(128, 128, 8, 4)
  PROBA9(64, 128, 4, 8)
#define PROBA10(BM, BN, TM, TN)                                               \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v10<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v10<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в10-1бар %3dx%-3d / %dx%d : %7.3f мс  %6.2f ТОпс\n",          \
             BM, BN, TM, TN, ms, tops);                                       \
      if (ms < best) { best = ms; bestName = "в10 " #BM "x" #BN "/" #TM "x" #TN; } \
    }                                                                         \
  }
  PROBA10(128, 64, 8, 4)
  PROBA10(64, 64, 4, 4)
  PROBA10(128, 128, 8, 8)
#define PROBA11(BM, BN, TM, TN)                                               \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v11<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v11<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в11-2групи %3dx%-3d / %dx%d : %7.3f мс  %6.2f ТОпс\n",        \
             BM, BN, TM, TN, ms, tops);                                       \
      if (ms < best) { best = ms; bestName = "в11 " #BM "x" #BN "/" #TM "x" #TN; } \
    }                                                                         \
  }
  PROBA11(128, 64, 8, 4)
  PROBA11(64, 64, 4, 4)
#define PROBA12(BM, BN, TM, TN)                                               \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v12<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v12<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в12-масштаби-в-LDS %3dx%-3d / %dx%d : %7.3f мс  %6.2f ТОпс\n",\
             BM, BN, TM, TN, ms, tops);                                       \
      if (ms < best) { best = ms; bestName = "в12 " #BM "x" #BN "/" #TM "x" #TN; } \
    }                                                                         \
  }
  PROBA12(128, 64, 8, 4)
  PROBA12(64, 64, 4, 4)
#define PROBA13(BM, BN, TM, TN)                                               \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        gemm_i4_v13<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        gemm_i4_v13<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K); \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  в13-пари %3dx%-3d / %dx%d : %7.3f мс  %6.2f ТОпс\n",          \
             BM, BN, TM, TN, ms, tops);                                       \
      if (ms < best) { best = ms; bestName = "в13 " #BM "x" #BN "/" #TM "x" #TN; } \
    }                                                                         \
  }
  PROBA13(128, 64, 8, 4)
  PROBA13(128, 64, 8, 8)
  PROBA13(64, 128, 8, 8)
  PROBA13(128, 64, 4, 8)
#define PROBAX(NAME, TAG, BM, BN, TM, TN)                                     \
  {                                                                           \
    dim3 blk(BM * BN / (TM * TN));                                            \
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);                           \
    if (blk.x <= 256 && blk.x >= 64) {                                        \
      for (int i = 0; i < 5; ++i)                                             \
        NAME<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);    \
      hipDeviceSynchronize();                                                 \
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);            \
      const int rep = 20; hipEventRecord(e0);                                 \
      for (int r = 0; r < rep; ++r)                                           \
        NAME<BM, BN, TM, TN><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);    \
      hipEventRecord(e1); hipEventSynchronize(e1);                            \
      float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= rep;              \
      double tops = 2.0 * M * N * K / (ms / 1e3) / 1e12;                      \
      printf("  %-22s %3dx%-3d / %dx%d : %7.3f мс  %6.2f ТОпс\n", TAG,        \
             BM, BN, TM, TN, ms, tops);                                       \
      if (ms < best) { best = ms; bestName = TAG; }                           \
    }                                                                         \
  }
  PROBAX(gemm_i4_v16, "в16-setprio", 128, 64, 8, 4)
  PROBAX(gemm_i4_v17, "в17-перестановка", 128, 64, 8, 4)
  {
    dim3 blk(128 * 64 / 32), grd((N + 63) / 64, (M + 127) / 128);
    for (int i = 0; i < 5; ++i)
      gemm_i4_diag3<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int r = 0; r < 20; ++r)
      gemm_i4_diag3<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 20;
    printf("  Д3 без арифметики масштабів (неправильно): %7.3f мс  %6.2f ТОпс\n",
           ms, 2.0 * M * N * K / (ms / 1e3) / 1e12);
  }
  {
    dim3 blk(128 * 64 / 32), grd((N + 63) / 64, (M + 127) / 128);
    for (int i = 0; i < 5; ++i)
      gemm_i4_diag4<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int r = 0; r < 20; ++r)
      gemm_i4_diag4<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 20;
    printf("  Д4 без читань LDS у циклі (неправильно):   %7.3f мс  %6.2f ТОпс\n",
           ms, 2.0 * M * N * K / (ms / 1e3) / 1e12);
  }
  {
    dim3 blk(128 * 64 / 32), grd((N + 63) / 64, (M + 127) / 128);
    for (int i = 0; i < 5; ++i)
      gemm_i4_diag5<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int r = 0; r < 20; ++r)
      gemm_i4_diag5<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 20;
    printf("  Д5 ні LDS, ні стягування (неправильно):    %7.3f мс  %6.2f ТОпс\n",
           ms, 2.0 * M * N * K / (ms / 1e3) / 1e12);
  }
  {
    dim3 blk(64 * 64 / 16), grd((N + 63) / 64, (M + 63) / 64);
    for (int i = 0; i < 5; ++i)
      gemm_i4_diag<64, 64, 4, 4, 1><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int r = 0; r < 20; ++r)
      gemm_i4_diag<64, 64, 4, 4, 1><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 20;
    printf("  Д1 без LDS І без стягування (неправильно): %7.3f мс  %6.2f ТОпс\n",
           ms, 2.0 * M * N * K / (ms / 1e3) / 1e12);
  }
  {
    dim3 blk(64 * 64 / 16), grd((N + 63) / 64, (M + 63) / 64);
    for (int i = 0; i < 5; ++i)
      gemm_i4_diag2<64, 64, 4, 4, 2><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int r = 0; r < 20; ++r)
      gemm_i4_diag2<64, 64, 4, 4, 2><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 20;
    printf("  Д2 стягування Є, читань LDS немає (неправильно): %7.3f мс  %6.2f ТОпс\n",
           ms, 2.0 * M * N * K / (ms / 1e3) / 1e12);
  }
  printf("  найкраще: %s  (%.3f мс, %.2f ТОпс)\n", bestName, best,
         2.0 * M * N * K / (best / 1e3) / 1e12);
  // Останній прогін найкращої форми лишає C для перевірки; беремо 128x128/8x8.
  {
    dim3 blk(128 * 64 / (8 * 4)), grd((N + 63) / 64, (M + 127) / 128);
    gemm_i4_v13<128, 64, 8, 4><<<grd, blk>>>(dA, dB, dAs, dBs, dC, M, N, K);
    hipDeviceSynchronize();
  }

  // ── Правильність: кілька випадкових елементів проти прямого рахунку ──
  std::vector<float> hC((long)M * N);
  hipMemcpy(hC.data(), dC, (long)M * N * 4, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int t = 0; t < 12; ++t) {
    int m = rand() % M, n = rand() % N;
    double ref = 0;
    for (int g = 0; g < NG; ++g) {
      long s = 0;
      for (int w = 0; w < WORDS; ++w) {
        int aw = hA[(long)m * K8 + g * WORDS + w];
        int bw = hB[(long)n * K8 + g * WORDS + w];
        for (int k = 0; k < 8; ++k) s += nibble(aw, k) * (long)nibble(bw, k);
      }
      ref += (double)s * hAs[(long)m * NG + g] * hBs[(long)n * NG + g];
    }
    double got = hC[(long)m * N + n];
    double vidn = fabs(got - ref) / (fabs(ref) + 1e-6);
    if (vidn > 1e-4) { ++bad; printf("  РОЗБІЖНІСТЬ m=%d n=%d: %.4f проти %.4f\n", m, n, got, ref); }
  }
  printf("  правильність: %s (12 випадкових елементів)\n",
         bad ? "ПОМИЛКА" : "збігається з прямим рахунком");
  return bad ? 1 : 0;
}
