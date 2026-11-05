// Самодостатній бенчмарк і перевірка ядра 8-бітного квантування активацій
// для AMD gfx1030 (RDNA2, wave32).
//
// Формат операндів відповідає схемі квантування ваг exllama з групою GS = 128:
// - Вхід: X [M, K] у fp16 (__half), суцільний
// - Виходи:
//     Aq [M, K/4] int32: по 4 беззнакові байти в слові
//     Su [M, K/128] float: суми беззнакових байтів по кожній групі
//     Sa [M, K/128] float: масштаби груп
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define GS 128

#define HIP_CHECK(expr)                                                       \
  do {                                                                        \
    hipError_t err = (expr);                                                  \
    if (err != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error at %s:%d: %s (%d)\n", __FILE__, __LINE__,    \
              hipGetErrorString(err), (int)err);                              \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

// Квантування активацій у 8 бітів з масштабом НА ГРУПУ зі 128 значень.
// Розкладка роботи: ОДНА ХВИЛЯ (32 смуги) на пару (рядок m, група g).
// У блоці 256 ниток = 8 хвиль.
__global__ __launch_bounds__(256) void quant_groups8(
    const __half* __restrict__ X,
    int* __restrict__ Aq,
    float* __restrict__ Su,
    float* __restrict__ Sa,
    int M, int K) {
  const int NG = K / GS;
  const int K4 = K / 4;
  const long wv = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (wv >= (long)M * NG) return;
  const int m = (int)(wv / NG), g = (int)(wv - (long)m * NG);
  const __half* base = X + (long)m * K + (long)g * GS;

  // 1. Максимум за модулем по групі.
  // Читання: кожна смуга бере 4 сусідні fp16 через два звертання до __half2.
  // Хвиля з 32 смуг суцільно покриває 128 елементів = 256 байтів.
  const __half2* b2 = (const __half2*)base;
  __half2 v0 = b2[lane * 2], v1 = b2[lane * 2 + 1];
  float mx = fmaxf(fmaxf(fabsf(__low2float(v0)), fabsf(__high2float(v0))),
                   fmaxf(fabsf(__low2float(v1)), fabsf(__high2float(v1))));
#pragma unroll
  for (int sft = 16; sft; sft >>= 1)
    mx = fmaxf(mx, __shfl_xor(mx, sft, 32));
  const float sa = fmaxf(mx, 1e-8f) / 127.0f;
  const float inv = 1.0f / sa;
  if (lane == 0) Sa[(long)m * NG + g] = sa;

  // 2. Пакування у 8 бітів і підсумовування Su.
  // Перемішування exllama: позиція p слова ваг w несе k = w*8 + shuf[p].
  // Одному слову ваг w відповідає пара слів активацій:
  // слово 2w несе shuf[0..3], слово 2w+1 несе shuf[4..7].
  // Смуга lane пакує слово активацій lane:
  // w = lane >> 1, половина half = lane & 1.
  const int shuf[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  const int w = lane >> 1;
  const int half = lane & 1;
  int word = 0;
  int lane_sum = 0;

#pragma unroll
  for (int b = 0; b < 4; ++b) {
    int k = w * 8 + shuf[half * 4 + b];
    float v = __half2float(base[k]) * inv;
    int q = (int)rintf(v) + 128;
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    word |= (q & 0xFF) << (8 * b);
    lane_sum += q;
  }
  Aq[(long)m * K4 + (long)g * 32 + lane] = word;

  int gs = lane_sum;
#pragma unroll
  for (int sft = 16; sft; sft >>= 1)
    gs += __shfl_xor(gs, sft, 32);
  if (lane == 0) Su[(long)m * NG + g] = (float)gs;
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int K = argc > 2 ? atoi(argv[2]) : 5120;
  if (K % GS != 0) {
    fprintf(stderr, "Помилка: K (%d) має ділитися на розмір групи GS (%d)\n", K, GS);
    return 1;
  }
  const int NG = K / GS;
  const int K4 = K / 4;

  // 1. Генерація випадкових X та обов'язкові канальні викиди
  srand(42);
  std::vector<__half> hX((size_t)M * K);
  for (size_t i = 0; i < (size_t)M * K; ++i) {
    float val = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
    hX[i] = __float2half(val);
  }

  // Канальні викиди: множимо 3 стовпці на 40-60
  int outlier_cols[3] = {42 % K, 1234 % K, 3500 % K};
  float outlier_mults[3] = {45.0f, 55.0f, 60.0f};
  for (int c = 0; c < 3; ++c) {
    int col = outlier_cols[c];
    float mult = outlier_mults[c];
    for (int r = 0; r < M; ++r) {
      float val = __half2float(hX[(size_t)r * K + col]);
      hX[(size_t)r * K + col] = __float2half(val * mult);
    }
  }

  const size_t x_bytes = (size_t)M * K * sizeof(__half);
  const size_t aq_bytes = (size_t)M * K4 * sizeof(int);
  const size_t su_bytes = (size_t)M * NG * sizeof(float);
  const size_t sa_bytes = (size_t)M * NG * sizeof(float);

  __half* dX = nullptr;
  int* dAq = nullptr;
  float* dSu = nullptr;
  float* dSa = nullptr;

  HIP_CHECK(hipMalloc(&dX, x_bytes));
  HIP_CHECK(hipMalloc(&dAq, aq_bytes));
  HIP_CHECK(hipMalloc(&dSu, su_bytes));
  HIP_CHECK(hipMalloc(&dSa, sa_bytes));

  HIP_CHECK(hipMemcpy(dX, hX.data(), x_bytes, hipMemcpyHostToDevice));

  // 2. Запуск ядра: 5 прогрівних, 20 замірюваних ітерацій
  const long hvyl = (long)M * NG;
  const int blokiv = (int)((hvyl + 7) / 8);

  for (int i = 0; i < 5; ++i) {
    quant_groups8<<<blokiv, 256>>>(dX, dAq, dSu, dSa, M, K);
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));

  const int NUM_RUNS = 20;
  HIP_CHECK(hipEventRecord(e0));
  for (int r = 0; r < NUM_RUNS; ++r) {
    quant_groups8<<<blokiv, 256>>>(dX, dAq, dSu, dSa, M, K);
  }
  HIP_CHECK(hipEventRecord(e1));
  HIP_CHECK(hipEventSynchronize(e1));

  float total_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&total_ms, e0, e1));
  float ms = total_ms / (float)NUM_RUNS;

  // Обчислення пропускної здатності:
  // Прочитано 2 байти на елемент X (fp16),
  // Записано 1 байт на елемент Aq (int32 / 4) + масштаби Sa (4 B) + суми Su (4 B) на групу.
  size_t bytes_read = (size_t)M * K * sizeof(__half);
  size_t bytes_written = (size_t)M * K4 * sizeof(int) + (size_t)M * NG * sizeof(float) * 2;
  size_t total_io_bytes = bytes_read + bytes_written;
  double bw_gbs = ((double)total_io_bytes / 1e9) / ((double)ms / 1000.0);

  printf("=== Квантування активацій у 8 бітів (quant_groups8) ===\n");
  printf("Конфігурація: M = %d, K = %d, GS = %d, NG = %d (всього груп: %ld)\n",
         M, K, GS, NG, (long)M * NG);
  printf("Час виконання (середній за %d ітерацій): %.4f мс\n", NUM_RUNS, ms);
  printf("Пропускна здатність: %.2f ГБ/с (прочитано %.2f МБ, записано %.2f МБ)\n\n",
         bw_gbs, (double)bytes_read / 1e6, (double)bytes_written / 1e6);

  // 3. ПЕРЕВІРКА на хості для 8 груп
  std::vector<int> hAq((size_t)M * K4);
  std::vector<float> hSu((size_t)M * NG);
  std::vector<float> hSa((size_t)M * NG);

  HIP_CHECK(hipMemcpy(hAq.data(), dAq, aq_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hSu.data(), dSu, su_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hSa.data(), dSa, sa_bytes, hipMemcpyDeviceToHost));

  printf("=== Перевірка коректності на хості (8 груп) ===\n");
  const int shuf[8] = {0, 2, 4, 6, 1, 3, 5, 7};

  struct GroupCoord { int m; int g; };
  std::vector<GroupCoord> test_groups;
  for (int c = 0; c < 3; ++c) {
    int col = outlier_cols[c];
    int g = col / GS;
    test_groups.push_back({0, g});
    test_groups.push_back({M / 2, g});
  }
  while (test_groups.size() < 8) {
    int rm = rand() % M;
    int rg = rand() % NG;
    test_groups.push_back({rm, rg});
  }

  bool all_match = true;
  for (size_t idx = 0; idx < 8; ++idx) {
    int m = test_groups[idx].m;
    int g = test_groups[idx].g;
    const __half* grp_x = hX.data() + (size_t)m * K + (size_t)g * GS;
    const int* grp_aq = hAq.data() + (size_t)m * K4 + (size_t)g * 32;

    // Еталонний розрахунок
    float ref_mx = 0.0f;
    for (int i = 0; i < GS; ++i) {
      ref_mx = fmaxf(ref_mx, fabsf(__half2float(grp_x[i])));
    }
    float ref_sa = fmaxf(ref_mx, 1e-8f) / 127.0f;
    float ref_inv = 1.0f / ref_sa;

    uint8_t ref_u[GS];
    int ref_su = 0;
    for (int i = 0; i < GS; ++i) {
      float v = __half2float(grp_x[i]) * ref_inv;
      int q = (int)rintf(v) + 128;
      q = q < 0 ? 0 : (q > 255 ? 255 : q);
      ref_u[i] = (uint8_t)q;
      ref_su += q;
    }

    float gpu_sa = hSa[(size_t)m * NG + g];
    if (fabsf(gpu_sa - ref_sa) > 1e-5f * fabsf(ref_sa)) {
      printf("ПОМИЛКА: група (m=%d, g=%d): Sa gpu=%.8f != ref=%.8f\n",
             m, g, gpu_sa, ref_sa);
      all_match = false;
    }

    float gpu_su = hSu[(size_t)m * NG + g];
    if (fabsf(gpu_su - (float)ref_su) > 1e-3f) {
      printf("ПОМИЛКА: група (m=%d, g=%d): Su gpu=%.1f != ref=%d\n",
             m, g, gpu_su, ref_su);
      all_match = false;
    }

    bool grp_bytes_ok = true;
    for (int w = 0; w < 16; ++w) {
      for (int j = 0; j < 8; ++j) {
        int word_idx = 2 * w + (j / 4);
        int byte_pos = j % 4;
        uint8_t gpu_byte = (uint8_t)((grp_aq[word_idx] >> (8 * byte_pos)) & 0xFF);
        int k = w * 8 + shuf[j];
        uint8_t expected_byte = ref_u[k];
        if (gpu_byte != expected_byte) {
          printf("ПОМИЛКА: група (m=%d, g=%d), w=%d, j=%d, k=%d: gpu_byte=%u != ref_u=%u\n",
                 m, g, w, j, k, gpu_byte, expected_byte);
          grp_bytes_ok = false;
          all_match = false;
          break;
        }
      }
      if (!grp_bytes_ok) break;
    }

    if (grp_bytes_ok && fabsf(gpu_sa - ref_sa) <= 1e-5f * fabsf(ref_sa) && fabsf(gpu_su - (float)ref_su) <= 1e-3f) {
      printf("  Група %zu (m=%4d, g=%2d): збігається з прямим рахунком (sa=%.6f, Su=%.0f)\n",
             idx, m, g, gpu_sa, gpu_su);
    }
  }

  if (all_match) {
    printf("Результат перевірки 8 груп: УСЕ ЗБІГАЄТЬСЯ З ПРЯМИМ РАХУНКОМ\n\n");
  } else {
    printf("Результат перевірки 8 груп: ПОМИЛКА звірки\n\n");
  }

  // 4. Відносна похибка розквантування по всій матриці
  printf("=== Відносна похибка розквантування по всій матриці ===\n");
  double norm_diff_sq = 0.0;
  double norm_orig_sq = 0.0;

  for (int m = 0; m < M; ++m) {
    for (int g = 0; g < NG; ++g) {
      float sa = hSa[(size_t)m * NG + g];
      const int* grp_aq = hAq.data() + (size_t)m * K4 + (size_t)g * 32;
      const __half* grp_x = hX.data() + (size_t)m * K + (size_t)g * GS;

      for (int w = 0; w < 16; ++w) {
        for (int j = 0; j < 8; ++j) {
          int word_idx = 2 * w + (j / 4);
          int byte_pos = j % 4;
          int u = (grp_aq[word_idx] >> (8 * byte_pos)) & 0xFF;
          int k = w * 8 + shuf[j];
          float x_orig = __half2float(grp_x[k]);
          float x_dequant = (float)(u - 128) * sa;
          double diff = (double)(x_dequant - x_orig);
          norm_diff_sq += diff * diff;
          norm_orig_sq += (double)x_orig * (double)x_orig;
        }
      }
    }
  }

  double rel_err = sqrt(norm_diff_sq) / sqrt(norm_orig_sq);
  printf("Відносна похибка розквантування ||(u-128)*sa - X|| / ||X||: %.4f%% (%.6e)\n",
         rel_err * 100.0, rel_err);

  HIP_CHECK(hipEventDestroy(e0));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipFree(dX));
  HIP_CHECK(hipFree(dAq));
  HIP_CHECK(hipFree(dSu));
  HIP_CHECK(hipFree(dSa));

  return all_match ? 0 : 1;
}
