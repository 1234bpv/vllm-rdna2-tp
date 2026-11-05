# SPDX-License-Identifier: Apache-2.0
"""All-reduce двох GPU через закріплену пам'ять хоста.

Потрібен на парі споживчих RDNA2 без взаємного P2P. Там RCCL мусить іти
транспортом SHM з проксі-потоками CPU, і найдрібніший обмін коштує ~150 мкс.
При тензорному паралелізмі це 130 обмінів на токен, тобто половина часу
декодування. Ядро в csrc/rocm/host_all_reduce.cu робить те саме за ~10 мкс.

Вмикається лише для дрібних обмінів: на великих RCCL швидший (див. таблицю
в шапці .cu). Вимкнути повністю: VLLM_ROCM_HOST_ALL_REDUCE=0.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Скільки останніх викликів тримати в кільцевому журналі. 32768 кроків з
# запасом перекривають вікно між розходженням лічильників і тайм-аутом.
_ROZMIR_ZHURNALU = 32768


class HostAllreduce:
    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        self.disabled = True
        self._max_elems = 0

        if not envs.VLLM_ROCM_HOST_ALL_REDUCE:
            return
        if not current_platform.is_rocm():
            return

        from vllm.platforms.rocm import on_gfx10

        if not on_gfx10():
            return

        world_size = dist.get_world_size(group=group)
        if world_size != 2:
            # Схема односкидна і розрахована рівно на два ранги.
            return

        rank = dist.get_rank(group=group)

        try:
            import vllm._custom_ops as ops  # noqa: F401

            self._ops = torch.ops._rocm_C
            # Ім'я сегмента спільне для обох рангів. Прив'язуємо до
            # ідентифікатора групи, щоб кілька двигунів на одній машині не
            # чіплялися до чужого сегмента.
            # Назву сегмента узгоджуємо через cpu-групу (gloo — на ньому
            # broadcast справний, на відміну від RCCL). Випадковий хвіст
            # потрібен, щоб після падіння й перезапуску не пристати до
            # сегмента попереднього запуску.
            tag = [f"{os.getuid()}_{os.getpid()}_{torch.randint(0, 2**31, (1,)).item()}"]
            dist.broadcast_object_list(tag, src=dist.get_global_rank(group, 0), group=group)
            name = f"/vllm_host_ar_{tag[0]}"
            self._ops.host_ar_init(rank, name)
            self._max_elems = int(self._ops.host_ar_max_elems())
        except Exception as e:  # noqa: BLE001
            logger.warning("Host all-reduce вимкнено: %s", e)
            self._agree(group)
            return

        self.disabled = False
        # ⚠️ Рішення мусить бути спільним для обох рангів. Якщо один піде цим
        # шляхом, а другий у RCCL, перший вічно чекатиме на епоху, якої ніхто
        # не опублікує, і обидві карти стануть намертво — а вийти з цього на
        # цій машині можна лише SIGKILL, тобто тим, чого робити не можна.
        self._agree(group)
        if self.disabled:
            try:
                self._ops.host_ar_destroy()
            except Exception:  # noqa: BLE001
                pass
            return
        self.rank = rank
        # ── Журнал викликів у пам'яті ─────────────────────────────────────
        # ⚠️ Пише В ПАМ'ЯТЬ і НІЧОГО не виводить, доки все гаразд. Це
        # принципово: попередні спроби діагностики зсували тайминг і збій
        # ЗНИКАВ — з увімкненим трасуванням 69 запитів проходили чисто, без
        # нього рушій падав на другому. Тут на виклик припадає додавання
        # кортежа в готовий список, тобто частки мікросекунди проти ~10 мкс
        # самого ядра. Вивалюємо на диск лише в мить розсинхронізації.
        self._zhurnal: list = [None] * _ROZMIR_ZHURNALU
        self._poz = 0
        logger.info(
            "Host all-reduce увімкнено (ранг %d, до %d елементів fp16). "
            "Між цими картами немає P2P, тож дрібні обміни через RCCL коштують "
            "~150 мкс; це ядро робить їх за ~10 мкс.",
            rank,
            self._max_elems,
        )

    def should_host_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        fp16 = inp.dtype == torch.float16
        sumizhnyj = inp.is_contiguous()
        elementiv = inp.numel()
        bere = fp16 and sumizhnyj and elementiv <= self._max_elems
        # Записуємо КОЖЕН виклик, і взятий, і ні: розходження рангів шукаємо
        # саме в послідовності рішень, тож пропущені виклики так само важливі.
        self._zhurnal[self._poz % _ROZMIR_ZHURNALU] = (
            self._poz, tuple(inp.shape), elementiv, fp16, sumizhnyj, bere,
        )
        self._poz += 1
        return bere

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(inp)
        try:
            self._ops.host_all_reduce(inp, out)
        except Exception:
            # Мить розсинхронізації — єдина, коли журнал має цінність.
            self._vyvantazhyty()
            raise
        return out

    def _vyvantazhyty(self) -> None:
        """Скинути кільцевий журнал на диск. Викликається лише при падінні."""
        shljah = os.path.join(
            envs.VLLM_ROCM_HOST_AR_DUMP_DIR,
            f"hostar-rank{self.rank}-{os.getpid()}.txt",
        )
        try:
            zapysy = [z for z in self._zhurnal if z is not None]
            zapysy.sort(key=lambda z: z[0])
            with open(shljah, "w", encoding="utf-8") as f:
                f.write(f"# ранг {self.rank}, викликів усього {self._poz}\n")
                f.write("# номер\tформа\tелементів\tfp16\tсуміжний\tчерез_ядро\n")
                for nomer, forma, n, fp16, sum_, bere in zapysy:
                    f.write(f"{nomer}\t{forma}\t{n}\t{int(fp16)}\t"
                            f"{int(sum_)}\t{int(bere)}\n")
            logger.error("host_all_reduce: журнал викликів збережено в %s", shljah)
        except Exception as e:  # noqa: BLE001
            logger.error("не вдалося зберегти журнал викликів: %s", e)

    def _agree(self, group) -> None:
        """Вимкнути в усіх рангах, якщо не вдалося бодай в одного."""
        import torch.distributed as dist

        flag = torch.tensor([0 if self.disabled else 1], dtype=torch.int32)
        try:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        except Exception:  # noqa: BLE001
            self.disabled = True
            return
        if int(flag.item()) == 0:
            if not self.disabled:
                logger.warning(
                    "Host all-reduce вимкнено: він не піднявся на іншому ранзі")
            self.disabled = True

    def close(self) -> None:
        if not self.disabled:
            try:
                self._ops.host_ar_destroy()
            except Exception:  # noqa: BLE001
                pass
            self.disabled = True
