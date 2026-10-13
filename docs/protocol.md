# Wire protocol

The LÖVE harness and the Python trainer talk over a TCP connection on
loopback. This page is the normative description; the two implementations are
`lua/rl_proto.lua` and `godhome_rl/env/proto.py`, which are kept in sync by
construction (both are short) and by the handshake check described below.

---

## Transport

* The **LÖVE process listens**, Python connects.
* `GODHOME_RL=port=<n>` picks the port (`0` = any free port). The harness prints
  `GODHOME_RL_READY port=<n>` on stdout and, if `port_file=<path>` is set, writes
  the number to that file.
* The harness prints the port **before** loading assets, so a boot failure
  arrives at Python as a readable message rather than a timeout.
* `TCP_NODELAY` is set on both ends. Socket timeouts are 30 s on the Lua side
  and 240 s on the Python side.

---

## Framing

```
+-----------------------+---------------------------+
| uint8 tag             | uint32 length (LE)       |
+-----------------------+---------------------------+
| payload[length]                                |
+-----------------------------------------------+
```

Header is 8 bytes. Lua packs it with the FFI and Python with `struct`.

---

## Tags

| Value | Name | Direction |
| --- | --- | --- |
| `0xA1` | `ACT` | Python → Lua |
| `0xA2` | `EPISODE` | Python → Lua |
| `0xA3` | `QUIT` | Python → Lua |
| `0xA4` | `PONG` | Python → Lua (reserved) |
| `0xB1` | `READY` | Lua → Python |
| `0xB2` | `STEP` | Lua → Python |
| `0xB3` | `LOG` | Lua → Python |
| `0xB4` | `PING` | Lua → Python (reserved) |

---

## Control messages (`ACT`, `EPISODE`, `QUIT`)

All three share one 32-byte payload matching the C struct in `rl_proto.lua`:

```c
typedef struct rl_msg_s {
    uint8_t  msg;        // offset 0
    /* 3 bytes padding */
    int32_t  action;     // offset 4
    int32_t  boss_index; // offset 8
    int32_t  seed;       // offset 12
    int32_t  max_steps;  // offset 16
    float    param0;     // offset 20   difficulty (1 attuned .. 4 radiant)
    float    param1;     // offset 24   1.0 = ascended
    float    param2;     // offset 28   reserved
} rl_msg_t;              // total 32 bytes, 4-byte aligned
```

Python: `struct.Struct("<Bxxxiiiifff")`.

| Message | Fields used |
| --- | --- |
| `ACT` | `action` |
| `EPISODE` | `boss_index` (or the trainer resolves it to an id), `seed`, `max_steps`, `param0`, `param1` |
| `QUIT` | none — the harness closes and exits |

`boss_id` is not on the wire as a string: the trainer resolves it against the
`bosses` list passed at launch, and the harness indexes that same list.

---

## `READY` — handshake

Payload is a flat array of little-endian `float32`, **in this exact order**:

| Index | Field |
| --- | --- |
| 0 | protocol version (`1`) |
| 1 | `obs_dim` |
| 2 | action count (`36`) |
| 3 | fixed timestep in seconds |
| 4 | `grid_w` |
| 5 | `grid_h` |
| 6 | `max_bullets` |
| 7 | global block size (`48`) |
| 8 | grid channels (`2`) |
| 9 | bullet feature count (`10`) |
| 10 | info vector length (`16`) |

Python calls `Layout.verify(handshake)` before every run and refuses to start
if any value disagrees with the Python-side constants. This is what stops a
stale `build/` from silently training on a differently shaped vector.

---

## `STEP` — observation

```
offset  size  field
0       4     uint32 obs_count
4       4     float32 reward
8       1     uint8  flags
9       1     uint8  info_count
10      2     padding
12      4*N   float32 obs[N]
12+4N   4*I   float32 info[I]
```

* `obs` is the raw observation vector — no packing, no quantisation.
* `reward` is the scalar for the frame that was just simulated.
* `info` is the 16-float diagnostics vector documented in
  [environment.md](environment.md).

### Flags

| Bit | Value | Meaning |
| --- | --- | --- |
| 0 | 1 | terminal (win or death) |
| 1 | 2 | truncated (frame budget exhausted) |
| 2 | 4 | episode boundary |
| 3 | 8 | first frame of an episode (reset) |

A reset frame always carries reward `0`: nothing has happened yet.

---

## `LOG`

UTF-8 text. Used for boot failures, "episode start failed" diagnostics, and — the
important one — the Lua traceback when the serve loop throws. Because the
harness sends the message *before* the socket closes, the reason for a crash
appears in the trainer's output instead of as a bare `ConnectionResetError`.

---

## Session shape

```
Lua                                          Python
──                                          ─────
listen(port)
write port file  ─────────────────────────►  (waits for the file)
accept()
                                            connect()
send READY           ─────────────────────►  verify layout
receive EPISODE      ◄─────────────────────   start_episode(spec)
start the fight,
advance to the
bullet-hell phase
send STEP(RESET)     ─────────────────────►  obs_0
receive ACT          ◄─────────────────────   choose a_0
step one frame
send STEP           ──────────────────────►  obs_1, r_1
receive ACT          ◄─────────────────────   choose a_1
...
send STEP(EPISODE)   ─────────────────────►   finish episode
receive EPISODE      ◄─────────────────────   choose next boss
...
receive QUIT         ◄─────────────────────   close()
close()
```

---

## Why not something fancier

* **TCP loopback** is available on every platform Godhome runs on, needs no
  extra dependency, and survives being called from a subprocess. LuaSocket is
  compiled into LÖVE ≥ 11.1, so there is nothing to install.
* **Uncompressed `float32`** keeps the Lua side free of an encoder. At 3.5 KiB
  per frame the bandwidth is irrelevant next to the simulation cost, and
  `np.frombuffer` on the Python side is a pointer cast rather than a parse.
* **The harness listens** so that Python controls the handshake, where a Python
  traceback is actually useful.