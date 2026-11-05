/*
 * All-reduce двох GPU через закріплену пам'ять хоста.
 *
 * Навіщо це існує. На парі споживчих RDNA2 без взаємного P2P
 * (hipDeviceCanAccessPeer == 0 в обидва боки) RCCL змушений іти транспортом
 * SHM: дані йдуть GPU -> пам'ять хоста -> GPU, а керують цим проксі-потоки
 * CPU. Через це найдрібніший обмін коштує ~150 мкс, хоча самих даних там на
 * частку мікросекунди. При тензорному паралелізмі це 130 обмінів на токен,
 * тобто ~54 мс, або половина всього часу декодування.
 *
 * Тут те саме робить одне ядро без участі CPU: кожен ранг пише свій вектор у
 * власний слот у спільній пам'яті хоста, оголошує номер кроку, чекає на
 * сусіда й додає чуже. Заміряно на gfx1030, два процеси:
 *
 *     елементів fp16     це ядро     RCCL
 *              5 120     10,3 мкс    ~150 мкс
 *             10 240     11,7 мкс    ~120 мкс
 *             40 960     26,3 мкс    ~133 мкс
 *            262 144    141,9 мкс    ~150 мкс   <- приблизно тут перевага зникає
 *
 * Тому ядро вмикається лише для дрібних обмінів; великі (префіл) лишаються за
 * RCCL, який на них швидший.
 *
 * Придатність до графів. Номер кроку тримається В ПАМ'ЯТІ, а не в аргументі
 * ядра: аргументи фіксуються під час захоплення графа, тож лічильник ззовні
 * лишався б сталим при кожному відтворенні. Дані подвійно буферизовані за
 * парністю кроку, тож ранг, що втік на крок уперед, не затирає те, чого сусід
 * ще не прочитав.
 */
#include <cstdlib>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <stdexcept>

namespace vllm {
namespace host_ar {

// Максимум елементів в одному слоті. 256 Кі елементів = 512 КіБ на буфер,
// 2 буфери на ранг, 2 ранги -> 2 МіБ спільної пам'яті. Це з запасом покриває
// зону, де ядро виграє в RCCL.
static constexpr int64_t kMaxElems = 256 * 1024;
static constexpr int kWorld = 2;

struct Slot {
  unsigned int epoch;      // скільки разів цей ранг увійшов у ядро
  unsigned int aborted;    // ядро здалося чекати на сусіда
  unsigned int pad[14];    // прапорець в окремому рядку кешу
  __half data[2][kMaxElems];
};
// Магічне слово наприкінці сегмента. Ранг 1 чекає саме на нього, а не просто
// на потрібний розмір: інакше він може почати працювати до того, як ранг 0
// обнулив пам'ять, і той затре йому лічильник кроку.
static constexpr unsigned int kMagic = 0x48415231u;  // "HAR1"

struct Shared {
  Slot slot[kWorld];
  volatile unsigned int magic;
};

// Стан на процес.
static Shared* g_host = nullptr;   // адреса на боці CPU
static Shared* g_dev = nullptr;    // та сама пам'ять очима GPU
static int g_rank = -1;
static std::string g_name;

template <typename scalar_t>
__device__ __forceinline__ scalar_t add_(scalar_t a, scalar_t b);
template <>
__device__ __forceinline__ __half add_(__half a, __half b) {
  return __hadd(a, b);
}

__global__ void ar_kernel(Slot* __restrict__ mine, const Slot* __restrict__ peer,
                          __half* __restrict__ out, const __half* __restrict__ in,
                          int n) {
  // ⚠️ Це ядро ЗОБОВ'ЯЗАНЕ запускатися одним блоком.
  //
  // Попередня версія йшла у 8 блоків, і це давало дві різні поломки, бо
  // __syncthreads() і __threadfence_system() діють у межах блока, а не сітки:
  //   * нульовий блок публікував епоху, записавши лише свою восьму частину
  //     даних — сусід читав решту з попередньої епохи (тихе псування чисел);
  //   * блок, що доходив до читання епохи ПІСЛЯ публікації, брав my_epoch на
  //     одиницю більше, обирав інший буфер і чекав на епоху, яка в цьому
  //     виклику вже не настане — вічне зависання ядра, тобто саме той стан,
  //     з якого на цій машині немає виходу без SIGKILL.
  // Один блок робить обидві поломки неможливими за побудовою. При n <= 256Ki
  // це все одно набагато дешевше за RCCL (10 мкс проти 150).
  __shared__ unsigned int my_epoch;
  int tid = threadIdx.x;
  int stride = blockDim.x;

  if (threadIdx.x == 0) my_epoch = ((volatile Slot*)mine)->epoch + 1;
  __syncthreads();
  int buf = my_epoch & 1;

  for (int i = tid; i < n; i += stride) mine->data[buf][i] = in[i];
  __threadfence_system();
  __syncthreads();

  __shared__ unsigned int s_aborted;
  __shared__ unsigned int s_timeout;  // ОКРЕМА змінна: див. нижче
  // Липкий чек НА ПРИСТРОЇ. Хостова перевірка нижче не виконується під
  // відтворенням CUDA-графа — а це саме шлях декоду, заради якого ядро й
  // існує. Без цього після тайм-ауту кожен наступний обмін кроку знову
  // спінив би 20 с (їх ~64), черга HSA стояла б десятки хвилин, і зупинка
  // сервера зводилася б до SIGKILL, тобто до тієї самої забороненої умови,
  // від якої тайм-аут і рятує.
  if (threadIdx.x == 0) s_aborted = ((volatile const Slot*)mine)->aborted;
  __syncthreads();
  if (s_aborted) {
    // Той самий NaN, що й у виклику, який здався (див. нижче): свій внесок
    // без сусіда — це половина суми, тобто тихо неправильне число.
    const __half otruta = __float2half(__builtin_nanf(""));
    for (int i = tid; i < n; i += stride) out[i] = otruta;
    return;
  }

  // ⚠️ Тайм-аут пише в ОКРЕМУ змінну, а не в s_aborted. Інакше між читанням
  // s_aborted усіма потоками (вище) і записом одиниці потоком 0 (нижче) не
  // було б бар'єра: хвиля, що запізнилася, прочитала б уже одиницю, пішла в
  // ранній return і не дійшла до __syncthreads() після циклу очікування, де
  // стоять інші. Це розбіжність на бар'єрі, тобто вічне зависання ядра —
  // саме той стан, з якого тут немає виходу без SIGKILL.
  if (threadIdx.x == 0) {
    s_timeout = 0u;
    ((volatile Slot*)mine)->epoch = my_epoch;
    __threadfence_system();

    // Обмежене очікування. Без межі смерть сусіда (а тут ~половина стартів
    // колись падала з hipErrorIllegalAddress) підвішувала це ядро назавжди,
    // черга HSA не спорожнялася, і охайна зупинка ставала неможливою.
    // ⚠️ ЗАМІРЯНО 10.09.2026: s_memrealtime іде рівно на 100 МГц, тобто такт
    // це 10 нс, а не 100. Стара формула 20000000000/100 давала 2·10⁸ тактів,
    // тобто межу в ДВІ секунди замість двадцяти — удесятеро коротшу за
    // задуману. Під агентним навантаженням карти тут різні (60 CU проти 72),
    // і відставання сусіда понад дві секунди буденне: рушій падав з
    // «Епохи розсинхронізовані» чотири рази за вечір, хоча журнали обох
    // рангів показували ІДЕНТИЧНУ послідовність з 32768 викликів.
    constexpr unsigned long long kTaktivNaSekundu = 100000000ULL;  // 100 МГц
    const unsigned long long kSpinLimit = 20ULL * kTaktivNaSekundu;  // 20 с
    unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
    // Порівнюємо ЗНАКОВОЮ різницею, а не «<»: епоха 32-бітна й раз на 2³²
    // обмінів переходить через нуль. При «peer < mine» ранг, що першим
    // перейшов на 0, бачив би 0xFFFFFFFF < 0 хибним і не чекав би сусіда,
    // підсумовуючи застарілий буфер. Різниця за модулем 2³² правильна, доки
    // ранги розходяться менш ніж на 2³¹ кроків, а тут вони розходяться
    // щонайбільше на один.
    while ((int)(((volatile const Slot*)peer)->epoch - my_epoch) < 0) {
      if (__builtin_amdgcn_s_memrealtime() - t0 > kSpinLimit) {
        s_timeout = 1u;
        ((volatile Slot*)mine)->aborted = 1u;
        __threadfence_system();
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  // Бар'єр набуття: без нього ніщо не забороняє підняти читання даних сусіда
  // вище очікування, а рядки кешу з позаминулої епохи лишаються придатними
  // (подвійна буферизація повторює буфер через раз).
  __threadfence_system();

  // Якщо очікування здалося, буфер сусіда містить дані з ПОЗАМИНУЛОЇ епохи
  // (подвійна буферизація повторює буфер через раз). Підсумовувати їх —
  // означає мовчки повернути неправильне число, тож не чіпаємо їх узагалі.
  // ⚠️ Під ВІДТВОРЕННЯМ CUDA-графа хостова перевірка в host_all_reduce() не
  // виконується взагалі. Тому прапорець експортовано як host_ar_aborted(), і
  // Python читає його після синхронізації відібраних токенів кроку.
  if (s_timeout) {
    // ⚠️ НЕ повертаємо свій внесок: для all-reduce це ПОЛОВИНА суми, тобто
    // так само неправильне число, просто неправильне інакше. Модель на цьому
    // не падає — вона спокійно генерує сміття (пробіли, крапки, двокрапки),
    // і під відтворенням CUDA-графа хостова перевірка мовчить, тож зіпсовані
    // токени йдуть десятками, поки не трапиться префіл. Пишемо NaN: він
    // отруює все далі за течією й унеможливлює тихе псування.
    const __half otruta = __float2half(__builtin_nanf(""));
    for (int i = tid; i < n; i += stride) out[i] = otruta;
    return;
  }
  const volatile __half* pdata = peer->data[buf];
  for (int i = tid; i < n; i += stride)
    out[i] = add_(in[i], const_cast<__half&>(pdata[i]));
}

}  // namespace host_ar
}  // namespace vllm

using namespace vllm::host_ar;

// Створити або приєднатися до спільного сегмента. Викликається один раз на ранг.
void host_ar_init(int64_t rank, const std::string& name) {
  TORCH_CHECK(rank >= 0 && rank < kWorld, "host_ar підтримує рівно 2 ранги");
  TORCH_CHECK(g_host == nullptr, "host_ar_init уже викликано");

  void* p = nullptr;
  if (rank == 0) {
    // Знімаємо стару назву: після падіння попереднього запуску сегмент
    // лишається з ненульовими лічильниками, і сусід міг би до нього пристати.
    shm_unlink(name.c_str());
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    TORCH_CHECK(fd >= 0, "shm_open('", name, "') не вдався: ", strerror(errno));
    TORCH_CHECK(ftruncate(fd, sizeof(Shared)) == 0, "ftruncate не вдався");
    p = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    TORCH_CHECK(p != MAP_FAILED, "mmap не вдався: ", strerror(errno));
    std::memset(p, 0, sizeof(Shared));
    __sync_synchronize();
    ((Shared*)p)->magic = kMagic;   // лише тепер сегмент готовий до вжитку
  } else {
    struct stat st;
    bool ready = false;
    for (int i = 0; i < 60000 && !ready; i++) {
      int fd = shm_open(name.c_str(), O_RDWR, 0600);
      if (fd >= 0) {
        if (fstat(fd, &st) == 0 && (size_t)st.st_size >= sizeof(Shared)) {
          void* q = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
          if (q != MAP_FAILED) {
            if (((Shared*)q)->magic == kMagic) {
              p = q;
              ready = true;
            } else {
              munmap(q, sizeof(Shared));
            }
          }
        }
        close(fd);
      }
      if (!ready) usleep(1000);
    }
    TORCH_CHECK(ready, "ранг 0 не підготував сегмент '", name, "' за 60 с");
  }

  // Зробити спільну пам'ять видимою для GPU саме цього процесу.
  auto err = hipHostRegister(p, sizeof(Shared), hipHostRegisterMapped);
  TORCH_CHECK(err == hipSuccess, "hipHostRegister не вдався: ", hipGetErrorString(err));
  void* dp = nullptr;
  err = hipHostGetDevicePointer(&dp, p, 0);
  TORCH_CHECK(err == hipSuccess, "hipHostGetDevicePointer не вдався: ",
              hipGetErrorString(err));

  g_host = (Shared*)p;
  g_dev = (Shared*)dp;
  g_rank = (int)rank;
  g_name = name;
}

void host_ar_destroy() {
  if (g_host == nullptr) return;
  hipHostUnregister(g_host);
  munmap(g_host, sizeof(Shared));
  if (g_rank == 0) shm_unlink(g_name.c_str());
  g_host = nullptr;
  g_dev = nullptr;
  g_rank = -1;
}

int64_t host_ar_max_elems() { return kMaxElems; }

bool host_ar_ready() { return g_dev != nullptr; }

// Чи зривався обмін. Читається з хоста (пам'ять спільна), тому дармово.
// Потрібно, бо під відтворенням CUDA-графа host_all_reduce() не викликається.
bool host_ar_aborted() {
  if (g_host == nullptr) return false;
  return ((volatile const Slot*)&g_host->slot[g_rank])->aborted != 0u;
}

// Сама операція. Вхід і вихід можуть збігатися.
void host_all_reduce(torch::Tensor& inp, torch::Tensor& out) {
  TORCH_CHECK(g_dev != nullptr, "host_ar_init не викликано");
  TORCH_CHECK(inp.scalar_type() == at::kHalf, "host_all_reduce поки лише fp16");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "потрібні суцільні тензори");
  int64_t n = inp.numel();
  TORCH_CHECK(n <= kMaxElems, "завелико для host_all_reduce: ", n, " > ", kMaxElems);
  // Ядро пише в out рівно n елементів __half: інший тип, розмір чи пристрій
  // означав би запис за межі тензора без жодної помилки.
  TORCH_CHECK(out.scalar_type() == at::kHalf, "host_all_reduce: out мусить бути fp16");
  TORCH_CHECK(out.numel() == n, "host_all_reduce: out має ", out.numel(),
              " елементів, а inp — ", n);
  TORCH_CHECK(out.device() == inp.device(),
              "host_all_reduce: inp і out на різних пристроях");

  const at::cuda::OptionalCUDAGuard guard(device_of(inp));
  auto stream = at::cuda::getCurrentCUDAStream();
  // Один блок — обов'язково, див. коментар у ядрі. Оскільки паралелізм між
  // блоками недоступний, лишається ширина блока: перебирається через
  // VLLM_ROCM_HOST_AR_THREADS, максимум для RDNA2 — 1024.
  static const int threads = [] {
    const char* e = std::getenv("VLLM_ROCM_HOST_AR_THREADS");
    int t = e ? std::atoi(e) : 1024;
    if (t < 64 || t > 1024) t = 1024;
    return (t / 64) * 64;
  }();
  // Прапорець із ПОПЕРЕДНЬОГО виклику. Слот лежить у пам'яті, спільній для
  // хоста й пристрою, тож читання дармове й не потребує синхронізації.
  // Якщо ядро колись здалося чекати, епохи розсинхронізовані назавжди й усі
  // подальші суми були б із застарілого буфера — падаємо голосно.
  TORCH_CHECK(((volatile Slot*)&g_host->slot[g_rank])->aborted == 0u,
              "host_all_reduce: ядро здалося чекати на сусіда (тайм-аут 20 с). "
              "Епохи розсинхронізовані, дальші результати були б неправильні. "
              "Найімовірніша причина — падіння процесу сусіднього ранга.");

  const int blocks = 1;
  hipLaunchKernelGGL(ar_kernel, dim3(blocks), dim3(threads), 0, stream,
                     &g_dev->slot[g_rank], &g_dev->slot[1 - g_rank],
                     (__half*)out.data_ptr(), (const __half*)inp.data_ptr(), (int)n);
}
