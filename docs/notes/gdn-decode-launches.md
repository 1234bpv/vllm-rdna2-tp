# Запуски ядер на кроці декоду в шарах GDN

Звіт підагента `agy` (gemini-3.7-flash-high) від 02.09.2026.
⚠️ Твердження підагента НЕ перевірені заміром — це карта для подальшої роботи,
а не результат. Ключове, що варто перевірити першим: чи справді крок декоду
упирається в запуски, адже лінійні шари вже на 84% смуги пам'яті.

Ось детальний звіт за результатами аналізу кодової бази `~/vllm`.

---

### 1. Перелік запусків на один крок декоду одного GDN-шару (шлях: 1 токен, стан у кеші)

На платформі ROCm для архітектури RDNA2 (gfx1030) обробка одного декод-токена в шарі GDN (`Qwen3NextDecoderLayer.forward`) проходить через `QwenGatedDeltaNetAttention`. 
Оскільки на gfx1030 AITER недоступний (`_aiter_ops.py:2007-2010`), `forward_hip` (`qwen_gdn_linear_attn.py:841-875`) перенаправляє виконання у `forward_cuda` (`qwen_gdn_linear_attn.py:877-951`).

Послідовність усіх окремих операцій, що дають запуск ядер (GPU kernel launch):

#### Блок 0: Нормалізація перед увагою (рівень шару декодера)
1. **ATen / Custom Kernel**: `input_layernorm` — `qwen3_next.py:533`, викликає `Qwen3_5RMSNorm` / `fused_add_rms_norm` (`layernorm.py:157-161`).

#### Блок 1: Вхідні лінійні проекції та підготовка тензорів
2. **ATen (GEMM / rocBLAS)**: `self.in_proj_qkvz(hidden_states)` — `qwen_gdn_linear_attn.py:891`. Проекція для Q, K, V та Z.
3. **ATen (GEMM / rocBLAS)**: `self.in_proj_ba(hidden_states)` — `qwen_gdn_linear_attn.py:892`. Проекція для $\beta$ та $\alpha$.
4. **ATen (zero_ / memset)**: `core_attn_out = torch.zeros(...)` — `qwen_gdn_linear_attn.py:936-940`. Ініціалізація вихідного буфера уваги.
5. **ATen (copy_)**: `b.contiguous()` — `qwen_gdn_linear_attn.py:944`. Викликається для результату `ba.chunk(2, dim=-1)` (`qwen_gdn_linear_attn.py:607`), бо зріз по останньому виміру не є неперервним.
6. **ATen (copy_)**: `a.contiguous()` — `qwen_gdn_linear_attn.py:945`. Аналогічно копіює зріз $\alpha$ в неперервний буфер.

#### Блок 2: Ядро уваги GDN (всередині `qwen_gdn_attention_core`)
Залежно від активного режиму декоду (штатний пакетний декод чи блочний декод через `VLLM_ROCM_GDN_DECODE_VIA_CHUNK`):

* **Варіант А: Штатний пакетний декод** (`qwen_gdn_linear_attn.py:1272-1288, 1717-1773`):
  7. **Triton Kernel**: `_causal_conv1d_update_kernel` — `causal_conv1d.py:1232`. Оновлює стан згортки `conv_state` і застосовує SiLU-активацію для 1 токена.
  8. **Triton Kernel**: `fused_recurrent_gated_delta_rule_packed_decode_kernel` — `fused_recurrent.py:456`. Оновлює рекурентний матричний стан $S_t$ (SSM cache) і обчислює вихід уваги з упакованого QKV-буфера.

* **Варіант Б: Блочний декод `VLLM_ROCM_GDN_DECODE_VIA_CHUNK=1`** (`qwen_gdn_linear_attn.py:1370-1398, 1563-1612`):
  7. **ATen (fill_)**: `has_init_decode = torch.ones(...)` — `qwen_gdn_linear_attn.py:1378`.
  8–9. **ATen / Triton**: `causal_conv1d_fn` — `qwen_gdn_linear_attn.py:1384-1398` (транспонування + ядро префіл-згортки + транспонування).
  10. **Triton Kernel**: `_fused_post_conv_kernel` — `fused_gdn_prefill_post_conv.py:20`.
  11. **ATen (index_select)**: `initial_state = ssm_state[decode_state_indices]` — `qwen_gdn_linear_attn.py:1594`.
  12. **Triton Kernel**: `chunk_local_cumsum` — `chunk.py:37`, `cumsum.py:130`.
  13. **Triton Kernel**: `chunk_scaled_dot_kkt_fwd` — `chunk.py:41`, `chunk_scaled_dot_kkt.py:110`.
  14. **Triton Kernel**: `solve_tril` — `chunk.py:49`, `solve_tril.py:88`.
  15. **Triton Kernel**: `recompute_w_u_fwd` — `chunk.py:52`, `wy_fast.py:138`.
  16. **Triton Kernel**: `chunk_gated_delta_rule_fwd_h` — `chunk.py:61`, `chunk_delta_h.py:45`.
  17. **Triton Kernel**: `chunk_fwd_o` — `chunk.py:72`, `chunk_o.py:82`.
  18. **ATen (index_copy_)**: `ssm_state[decode_state_indices] = ...` — `qwen_gdn_linear_attn.py:1612`.
  19. **ATen (copy_)**: `core_attn_out[:num_actual_tokens] = ...` — `qwen_gdn_linear_attn.py:1649`.

* **Варіант В: Непакетний рекурентний шлях** (`enable_packed_recurrent_decode=0`, `qwen_gdn_linear_attn.py:1401-1649`):
  7. **Triton Kernel**: `_causal_conv1d_update_kernel` — `causal_conv1d.py:1232`.
  8. **ATen (cat / copy_)**: `torch.cat([query..., key..., value...])` у `rearrange_mixed_qkv` — `qwen_gdn_linear_attn.py:799-801`.
  9–13. **ATen (copy_)**: 5 викликів `.contiguous()` для `a, b, q, k, v` у `fused_sigmoid_gating_delta_rule_update` — `fused_sigmoid_gating.py:244-251`.
  14. **Triton Kernel**: `fused_sigmoid_gating_delta_rule_update_kernel` — `fused_sigmoid_gating.py:242`.
  15. **ATen (copy_)**: `core_attn_out[...] = ...` — `qwen_gdn_linear_attn.py:1649`.

#### Блок 3: Вихідне нормування та проекція
(Номери нижче наведено для Варіанта А / Варіанта Б):
* **Triton Kernel**: `layer_norm_fwd` (`RMSNormGated`) — `layernorm_guard.py:287`, викликається через `_output_projection` (`qwen_gdn_linear_attn.py:835`). Обчислює RMSNorm над виходом уваги з SiLU-гейтингом від вектора $Z$.
* **ATen (GEMM / rocBLAS)**: `self.out_proj` — `qwen_gdn_linear_attn.py:838`. Лінійна проекція виходу уваги (+ All-Reduce при TP > 1).

#### Блок 4: Нормалізація та MLP шару декодера
* **ATen / Custom Kernel**: `post_attention_layernorm` — `qwen3_next.py:573`.
* **ATen (GEMM / rocBLAS)**: `mlp.gate_up_proj` — `qwen2_moe.py:65`.
* **Triton / Custom Kernel**: `silu_and_mul` — `qwen2_moe.py:72`.
* **ATen (GEMM / rocBLAS)**: `mlp.down_proj` — `qwen2_moe.py:73` (+ All-Reduce при TP > 1).

---

### 2. Кількість запусків на шар і на всю модель

У моделі Qwen3.5-27B всього **64 шари**:
* **48 GDN-шарів** (`layer_idx % 4 != 0`, `qwen3_5.py:255`).
* **16 шарів повної уваги** (`layer_idx % 4 == 0`).

Розрахунок запусків для одного кроку декоду:

| Складова кроку | Запусків на шар (Варіант А: штатний пакетний) | Запусків на шар (Варіант Б: `DECODE_VIA_CHUNK=1`) | Запусків на шар (Варіант В: розпакований) |
| :--- | :---: | :---: | :---: |
| **Увага GDN (лінійні + ядра + копії)** | 8 | 19–21 | 14–16 |
| **Норми та MLP декодерного шару** | 5 | 5 | 5 |
| **Всього на один GDN-шар** | **13–14** | **24–26** | **19–21** |
| **48 GDN-шарів (разом)** | **~624–672** | **~1152–1248** | **~912–1008** |
| **16 Full-Attention шарів (~12 запусків/шар)** | ~192 | ~192 | ~192 |
| **Ембединг, кінцевий RMSNorm, LM Head, семплер** | ~20–30 | ~20–30 | ~20–30 |
| **Загалом на крок декоду моделі** | **~850–900** | **~1360–1470** (до **~1650** з дрібними алокаціями/зрізами) | **~1120–1230** |

> [!NOTE]
> Число **~1650 запусків ядер** точно відповідає випадку, коли увімкнено блочний декод `VLLM_ROCM_GDN_DECODE_VIA_CHUNK=1` (який додає по 11 дрібних ядер на кожен із 48 GDN-шарів) разом із супутніми non-contiguous копіями та комунікаційними ядрами All-Reduce.

---

### 3. Що з цього дрібне й зливається (можливості оптимізації)

1. **Злиття виділення та contiguous-копій $b, a$ і обнулення виходу**
   * **Рядки**: `qwen_gdn_linear_attn.py:936-945`.
   * **Що відбувається**: виклик `torch.zeros(...)` ініціалізує буфер, а `b.contiguous()` та `a.contiguous()` виконують два окремих ядра D2D-копіювання для усунення strided-розриву після `ba.chunk(2)`.
   * **Що зливається**: передавати страйди або зчитувати `ba` безпосередньо у ядрі уваги; виділяти `torch.empty` замість `torch.zeros`, оскільки ядро уваги повністю перезаписує вихід.
   * **Економія**: **3 запуски на шар** (144 запуски на 48 GDN-шарів).
   * **Арифметика**: **100% без змін** (біт-у-біт).

2. **Злиття оновлення 1D-згортки та рекурентного оновлення уваги (Conv1D + Recurrent Decode)**
   * **Рядки**: `qwen_gdn_linear_attn.py:1751-1772` (`causal_conv1d_update` $\to$ `fused_recurrent_gated_delta_rule_packed_decode`).
   * **Що відбувається**: для $T=1$ проміжний результат `mixed_qkv_non_spec` записується ядром згортки у VRAM, а потім негайно зчитується наступним ядром рекурентної уваги.
   * **Що зливається**: об'єднання `causal_conv1d_update` безпосередньо в Triton-ядро декоду (зчитування `conv_state`, обчислення згортки в регістрах/LDS, запис нового `conv_state` і перехід до оновлення $S_t$).
   * **Економія**: **1 запуск на шар** (48 запусків на модель) + економія смуги пам'яті на проміжному тензорі.
   * **Арифметика**: **100% без змін**.

3. **Злиття виходу рекурентної уваги з `RMSNormGated`**
   * **Рядки**: `qwen_gdn_linear_attn.py:835`, `layernorm_guard.py:287-298`.
   * **Що відбувається**: ядро уваги пише ненормований вектор `core_attn_out` у глобальну пам'ять $\to$ окреме Triton-ядро `layer_norm_fwd` зчитує `core_attn_out` та `z`, рахує RMSNorm + SiLU gating.
   * **Що зливається**: нормалізація та гейтинг на векторі $Z$ виконуються на виході рекурентного ядра перед записом у глобальну пам'ять (аналогічно до підходу в `fused_gdn_decode_kernel.cu`).
   * **Економія**: **1 запуск на шар** (48 запусків на модель).
   * **Арифметика**: **100% без змін**.

4. **Усунення ланцюжка `torch.split` / `torch.cat` / `.contiguous()` у розпакованому шляху**
   * **Рядки**: `qwen_gdn_linear_attn.py:797-809`, `fused_sigmoid_gating.py:244-251`.
   * **Що відбувається**: розбиття упакованого QKV на 3 частини, зшивання їх через `torch.cat` в один буфер і наступні 5 викликів `.contiguous()`.
   * **Що зливається**: використання повністю упакованого формату (як у `fused_recurrent_gated_delta_rule_packed_decode`).
   * **Економія**: **6–7 запусків на шар** (~288–336 запусків на модель).
   * **Арифметика**: **100% без змін**.

5. **Відмова від блочних ядер `chunk_*` на кроці декоду ($T=1$)**
   * **Рядки**: `qwen_gdn_linear_attn.py:1563-1612`, `chunk.py:37-82`.
   * **Що відбувається**: запуск 6 окремих Triton-ядер (`chunk_local_cumsum`, `chunk_scaled_dot_kkt_fwd`, `solve_tril`, `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_h`, `chunk_fwd_o`) та 5 операцій копіювання/індексації замість одного рекурентного кроку.
   * **Що зливається**: усунення розходження в арифметиці між покроковим рекурентним ядром та блочним ядром (шляхом узгодження накопичувача FP32 та порядку операцій) з переведенням декоду на єдине покрокове рекурентне ядро.
   * **Економія**: **9–11 запусків на шар** (432–528 запусків на модель).

---

### 4. Чи є вже готові злиті шляхи

У кодовій базі вже реалізовано кілька злитих варіантів, але більшість із них не діють на gfx1030 для Qwen3.5:

1. **`fused_recurrent_gated_delta_rule_packed_decode`** (`fused_recurrent.py:343-480`)
   * **Статус**: **Працює на Triton/ROCm**. Зливає роботу з упакованим `mixed_qkv` без розділення на $q, k, v$. Вмикається прапорцем `envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` (`qwen_gdn_linear_attn.py:491-493`).

2. **`fused_gdn_decode_kernel.cu` / `qwen_gdn_attention_core_fused_norm_packed`** (`fused_gdn_decode_kernel.cu:1`, `qwen_gdn_linear_attn.py:894-912, 1856-1885`)
   * **Статус**: **Недоступно на ROCm**. Повністю монолітне C++/CUDA ядро (згортка + рекурентна увага + RMSNormGated в 1 запуск). Заблоковане перевіркою `current_platform.has_device_capability(80)` та відсутністю компіляції під HIP (`qwen_gdn_linear_attn.py:526, 534`).

3. **AITER Triton декод ядра (`gdn_aiter_fused_*`)** (`qwen_gdn_linear_attn.py:81-88, 1677-1715`)
   * **Статус**: **Недоступно на RDNA2 gfx1030**. Призначено лише для CDNA (gfx9) або RDNA4 (gfx12) (`_aiter_ops.py:1803-1814, 2007-2010`) та вимагає перемежованого GQA-лейауту Qwen3-Next (`qwen_gdn_linear_attn.py:1220`), тоді як Qwen3.5 має неперемежований лейаут.

4. **`fused_post_conv_prep`** (`fused_gdn_prefill_post_conv.py:20`)
   * **Статус**: Готове Triton-ядро під префіл (`split -> rearrange -> contiguous*3 -> l2norm*2 -> gating`), але на декоді викликається тільки якщо увімкнено `VLLM_ROCM_GDN_DECODE_VIA_CHUNK=1` (`qwen_gdn_linear_attn.py:1582`).

5. **Злиття проекцій у чекпоінті (`in_proj_qkvz`, `in_proj_ba`)** (`qwen3_5.py:220-227, 307-308`)
   * **Статус**: **Працює**. Зливає 6 окремих лінійних проекцій у 2 GEMM-виклики.

---

### 5. Чи заважає `torch.compile` (чи може Inductor злити це автоматично)

**Ні, `torch.compile` НЕ може злити ці операції автоматично.** Ручне злиття є абсолютно критичним і необхідним.

Причини в коді:
1. **Непрозорий Custom Op**: увага GDN зареєстрована у диспетчері PyTorch через `direct_register_custom_op` (`qwen_gdn_linear_attn.py:2033-2038`) з абстрактною заглушкою `gdn_attention_core_fake` (`qwen_gdn_linear_attn.py:2021-2030`).
2. **Межа графа (Graph Boundary)**: Torch Dynamo трасує лише зовнішню оболонку `Qwen3_5Model` (`qwen3_5.py:207`), де `torch.ops.vllm.qwen_gdn_attention_core` розглядається як «чорна скринька». Тіло функції `_forward_core` (`qwen_gdn_linear_attn.py:1246`) виконується у звичайному режимі **PyTorch Eager**.
3. **Блокування компіляції Triton-ядер**: `chunk_gated_delta_rule` має прямий декоратор `@torch.compiler.disable` (`chunk.py:137`).
4. **Виклик Triton поза Inductor**: виклики `RMSNormGated` напряму викликають `@triton.jit` ядро `layer_norm_fwd` (`layernorm_guard.py:287`). PyTorch Inductor не вміє інлайнити чи зливати рукописні Triton-ядра з сусідніми ATen-операціями.

Отже, жодна проміжна операція всередині `qwen_gdn_attention_core` не бачиться компілятором Inductor, і всі ланцюжки `contiguous()`, `zeros()`, `causal_conv1d_update` та рекурентної уваги виконуються як окремі дрібні GPU-ядра.

---

### Чого не встановити читанням і що для цього треба запустити

1. **Точний активний стан середовища під час заміру**: чи було встановлено змінну `VLLM_ROCM_GDN_DECODE_VIA_CHUNK=1` під час профілювання (що дає ~1650 запусків), чи замір проводився на стандартному `packed_decode`.
   * *Що запустити*: одноразовий запуск без GPU або перевірка логів vLLM: `logger.info_once("GDN decode kernel...")` та стан `envs.VLLM_ROCM_GDN_DECODE_VIA_CHUNK`.
2. **Точний профіль затримок кожного дрібного ядра**: який саме відсоток від 40 мс кроку декоду займає латентність диспетчеризації HIP API (launch latency ~3–5 мкс на ядро) порівняно з часом виконання самого коду на Compute Units.
   * *Що запустити (після звільнення карт)*: `rocprof --hip-trace --stats` або вбудований PyTorch Profiler на 10 кроках генерації декоду.
3. **Причина розходження точності FP32/BF16 між `fused_recurrent` і `chunk` на RDNA2**: чому саме покроковий шлях давав 10.6% розходження токенів (про що йдеться у коментарі від 29.08 на `qwen_gdn_linear_attn.py:1567`).
   * *Що запустити (після звільнення карт)*: порівняльний тест виходів тензорів $S_t$ між `fused_recurrent_gated_delta_rule_packed_decode` та `chunk_gated_delta_rule` на синтетичних входах для ізоляції помилки накопичення в інструкціях `v_fma_f32`.

[exited with code 0]
