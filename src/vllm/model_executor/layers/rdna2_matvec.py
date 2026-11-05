# SPDX-License-Identifier: Apache-2.0
"""Матвектор для неквантованих та int8-квантованих лінійних шарів на RDNA2 (gfx103x).

Навіщо. У Qwen3.5-27B-AWQ `lm_head` не квантований (він у списку `ignore`
чекпоінта), тобто це 248320x5120 у fp16 — 1,18 ГіБ на карту при TP=2. На кожен
крок декоду він читається цілком. hipBLASLt на gfx1030 бере для нього тайл,
розрахований на великі M, і при M=1 витрачає 16,4 мс замість 3,0 мс, які дає
пропускна здатність пам'яті. Це виходило ~22% усього кроку декодування.

Звідки це ядро. Штатні скіні-ядра vLLM (`wvSplitK`, `LLMM1`) на gfx1030 спершу
давали неправильний результат (похибка 0,46 і 0,62 на формі lm_head). Ваду
`wvSplitK` (запуск із шириною 64 замість хвилі 32) з 19.08 виправлено в
`csrc/rocm/skinny_gemms.cu`, і тепер при n <= 5 у
`layers/utils.py:rocm_unquantized_gemm_impl` першим іде саме він (2,82 мс
проти 3,39 мс тут). Це ядро лишається для того, що `wvSplitK` не бере
(скінні-шлях вимкнено, n > 5 у межах MAX_ROWS), а його int8-варіант — бойовий
шлях lm_head. `LLMM1` на gfx10 досі вимкнено: у нього окрема вада зведення.

Заміряно на RX 6800, форма 124160x5120, fp16:

    hipBLASLt  16,44 мс
    це ядро     3,38 мс   (4,9x; дах пам'яті — 3,02 мс, тобто взято 89%)
    похибка проти еталона 3,3e-04 (накопичення у fp32)

W8 шлях для lm_head (VLLM_ROCM_LMHEAD_INT8):
    Ваги int8 [M, K], масштаби рядків float32 [M], активації fp16 [N, K].
    Накопичення у tl.float32, множення на масштаб на рівні акумулятора,
    вихідний тензор повертається у torch.float32 (усуває крок сітки FP16 2^-7).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _matvec_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_CTX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros((N_CTX, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # W має вигляд [M, K] рядками, тож читання вздовж K суцільне.
        w = tl.load(
            w_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        for n in tl.static_range(N_CTX):
            # ⚠️ N_CTX — це next_pow2(N), тож при N = 3, 5, 6, 7 цикл проходить
            # більше рядків, ніж їх є. Без `n < N` це читання за межами x і
            # (нижче) запис за межами out: при M = 124160 у lm_head — чверть
            # мегабайта чужої пам'яті. Найімовірніше джерело періодичного
            # hipErrorIllegalAddress.
            xv = tl.load(
                x_ptr + n * K + offs_k, mask=mask_k & (n < N), other=0.0
            ).to(tl.float32)
            acc = tl.where(
                tl.arange(0, N_CTX)[:, None] == n,
                acc + tl.sum(w * xv[None, :], axis=1)[None, :],
                acc,
            )
    for n in tl.static_range(N_CTX):
        vals = tl.sum(tl.where(tl.arange(0, N_CTX)[:, None] == n, acc, 0.0), axis=0)
        tl.store(
            o_ptr + n * M + offs_m,
            vals.to(o_ptr.dtype.element_ty),
            mask=mask_m & (n < N),
        )


@triton.jit
def _matvec_int8_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    o_ptr,
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_CTX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros((N_CTX, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # W має вигляд [M, K] рядками (int8), читання вздовж K суцільне.
        w = tl.load(
            w_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.float32)
        for n in tl.static_range(N_CTX):
            xv = tl.load(
                x_ptr + n * K + offs_k, mask=mask_k & (n < N), other=0.0
            ).to(tl.float32)
            acc = tl.where(
                tl.arange(0, N_CTX)[:, None] == n,
                acc + tl.sum(w * xv[None, :], axis=1)[None, :],
                acc,
            )
    # Масштаб рядка [M] у float32
    scale = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    for n in tl.static_range(N_CTX):
        raw_val = tl.sum(
            tl.where(tl.arange(0, N_CTX)[:, None] == n, acc, 0.0), axis=0
        )
        # Множення на масштаб рядка НА РІВНІ АКУМУЛЯТОРА
        vals = raw_val * scale
        # ⚠️ Вихід у float32 без урізання до fp16
        tl.store(
            o_ptr + n * M + offs_m,
            vals.to(tl.float32),
            mask=mask_m & (n < N),
        )


# Понад цю кількість рядків тайловий GEMM уже виграє, тож ядро не застосовуємо.
MAX_ROWS = 8
# З VLLM_ROCM_SMALLN_FIX ядра беруть до 16 рядків: крок перевірки MTP має
# потоки × 3 рядків, і на 9..12 «широкий випадок» коштував 33 мс.
MAX_ROWS_SMALLN = 16


def max_rows() -> int:
    from vllm import envs

    return MAX_ROWS_SMALLN if envs.VLLM_ROCM_SMALLN_FIX > 0 else MAX_ROWS


def rdna2_matvec(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """x [n, K] @ weight[M, K]^T -> [n, M]. Вимагає суцільних тензорів."""
    n, k = x.shape
    m = weight.shape[0]
    out = torch.empty((n, m), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(m, 64),)
    # Понад MAX_ROWS — шматки по 8 рядків (лише з VLLM_ROCM_SMALLN_FIX):
    # заміряно, що два виклики N_CTX=8 швидші за один N_CTX=16 (int8 lm_head
    # 4,0 проти 5,2 мс на n=9..12). Шматки суцільні, форми статичні.
    for i in range(0, n, MAX_ROWS):
        nn = min(MAX_ROWS, n - i)
        _matvec_kernel[grid](
            x[i : i + nn],
            weight,
            out[i : i + nn],
            m,
            k,
            nn,
            BLOCK_M=64,
            BLOCK_K=128,
            N_CTX=triton.next_power_of_2(nn),
            num_warps=4,
        )
    if bias is not None:
        out += bias
    return out


def rdna2_matvec_int8(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """x [n, K] (fp16) @ (weight_int8[M, K] * scale[M])^T -> [n, M] у float32.

    Вимагає суцільних тензорів:
      - x: [n, K], torch.float16
      - weight_int8: [M, K], torch.int8
      - weight_scale: [M], torch.float32
      - повертає: [n, M], torch.float32
    """
    n, k = x.shape
    m = weight_int8.shape[0]
    out = torch.empty((n, m), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, 64),)
    for i in range(0, n, MAX_ROWS):  # шматки по 8: див. rdna2_matvec
        nn = min(MAX_ROWS, n - i)
        _matvec_int8_kernel[grid](
            x[i : i + nn],
            weight_int8,
            weight_scale,
            out[i : i + nn],
            m,
            k,
            nn,
            BLOCK_M=64,
            BLOCK_K=128,
            N_CTX=triton.next_power_of_2(nn),
            num_warps=4,
        )
    if bias is not None:
        out += bias.to(torch.float32)
    return out


def should_use_rdna2_matvec(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if x.dim() != 2:
        return False
    n, k = x.shape
    if n > max_rows() or n < 1:
        return False
    # Лише fp16: на RDNA2 немає `v_dot2_bf16_bf16`, тож bf16 тут непридатний
    # (сервер і так іде з --dtype float16, але шлюз не має бути ширшим за
    # платформу — інакше він мовчки пропустить непрацездатний тип).
    if x.dtype is not torch.float16:
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    if weight.shape[1] != k:
        return False
    # Поріг низький свідомо: у GDN-шарах є дрібні неквантовані проєкції
    # (~1,4 МБ ваг), і hipBLASLt витрачає на них 68 мкс при даху ~3 мкс.
    return weight.shape[0] * k >= 64 * 1024


def should_use_rdna2_matvec_int8(
    x: torch.Tensor, weight_int8: torch.Tensor
) -> bool:
    if x.dim() != 2:
        return False
    n, k = x.shape
    if n > max_rows() or n < 1:
        return False
    if x.dtype is not torch.float16:
        return False
    if not (x.is_contiguous() and weight_int8.is_contiguous()):
        return False
    if weight_int8.shape[1] != k:
        return False
    if weight_int8.dtype is not torch.int8:
        return False
    return weight_int8.shape[0] * k >= 64 * 1024
