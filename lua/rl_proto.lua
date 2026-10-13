-- ============================================================================
--  rl_proto.lua -- binary wire protocol between the LÖVE process and Python.
--
--  Design notes
--  ------------
--  * LuaJIT's FFI does the packing; `string.pack` does not exist in 5.1.
--  * Every message is  [uint8 tag][uint32 payload_len][payload bytes].
--    A single `recv(n)` per message keeps framing trivial and lets us use
--    plain blocking LuaSocket calls (no select() polling).
--  * Payloads are raw little-endian records; see `godhome_rl/env/proto.py`
--    for the mirrored struct definitions.  Both sides must agree.
-- ============================================================================

local ffi = require("ffi")

local Proto = {}

ffi.cdef[[
typedef struct rl_hdr_s {
    uint8_t  tag;
    uint8_t  _pad[3];
    uint32_t len;
} rl_hdr_t;

typedef struct rl_msg_s {
    uint8_t  msg;
    int32_t  action;
    int32_t  boss_index;
    int32_t  seed;
    int32_t  max_steps;
    float    param0;
    float    param1;
    float    param2;
} rl_msg_t;
]]

Proto.MSG_ACT    = 0xA1   -- Python -> Lua: current action for the next step
Proto.MSG_EPISODE= 0xA2   -- Python -> Lua: start a new episode with this spec
Proto.MSG_QUIT   = 0xA3   -- Python -> Lua: shut down
Proto.MSG_PONG   = 0xA4   -- Python -> Lua: reply to a request

Proto.MSG_READY  = 0xB1   -- Lua -> Python: harness is up
Proto.MSG_STEP   = 0xB2   -- Lua -> Python: observation / reward / flags / info
Proto.MSG_LOG    = 0xB3   -- Lua -> Python: free-form log line
Proto.MSG_PING   = 0xB4   -- Lua -> Python: liveness probe

-- flags inside MSG_STEP
Proto.FLAG_TERMINAL  = 1
Proto.FLAG_TRUNCATED = 2
Proto.FLAG_EPISODE   = 4
Proto.FLAG_RESET     = 8

local HDR_SZ = 8
Proto.HDR_SZ = HDR_SZ

--------------------------------------------------------------------------------
-- Reusable scratch buffers (avoid per-step allocations).
--------------------------------------------------------------------------------

local MAX_STEP_FLOATS = 65536

local _hdr    = ffi.new("rl_hdr_t[1]")
local _msgbuf = ffi.new("uint8_t[64]")
local _step   = ffi.new("float[?]", MAX_STEP_FLOATS)

Proto.scratch = _step
Proto.scratch_capacity = MAX_STEP_FLOATS

local function pack_u32(buf, off, v)
    ffi.cast("uint32_t*", buf + off)[0] = v
end

--------------------------------------------------------------------------------
-- Sending
--------------------------------------------------------------------------------

local function send_msg(client, tag, payload, len)
    _hdr[0].tag = tag
    _hdr[0].len = len or 0
    local ok, err = client:send(ffi.string(_hdr, HDR_SZ))
    if not ok then return nil, err end
    if len and len > 0 then
        ok, err = client:send(payload)
        if not ok then return nil, err end
    end
    return true
end

--- Send the capability handshake.  `values` must be an *ordered* array of
--  numbers: hash-table iteration order is not defined in Lua.
function Proto.send_ready(client, values)
    local n = #values
    if n == 0 then
        return send_msg(client, Proto.MSG_READY, nil, 0)
    end
    local floats = ffi.new("float[?]", n)
    for i = 1, n do
        floats[i - 1] = values[i]
    end
    return send_msg(client, Proto.MSG_READY, ffi.string(floats, n * 4), n * 4)
end

--- Push one observation frame.
--  @param floats   ffi float array (must stay alive until send completes)
--  @param n        number of floats in `obs`
--  @param reward   scalar
--  @param flags    bitfield
--  @param info     float array or nil
--  @param info_len number of floats in `info`
function Proto.send_step(client, floats, n, reward, flags, info, info_len)
    info_len = info_len or 0
    local body = 12 + (n * 4) + (info_len * 4)
    _hdr[0].tag = Proto.MSG_STEP
    _hdr[0].len = body
    -- header (12 bytes): uint32 count, float reward, uint8 flags,
    --                     uint8 info_len, uint16 padding
    local u = ffi.cast("uint8_t*", _msgbuf)
    ffi.cast("uint32_t*", _msgbuf)[0] = n
    ffi.cast("float*", _msgbuf + 4)[0] = reward
    u[8] = flags
    u[9] = info_len
    u[10] = 0
    u[11] = 0

    local ok, err = client:send(ffi.string(_hdr, HDR_SZ))
    if not ok then return nil, err end
    ok, err = client:send(ffi.string(_msgbuf, 12))
    if not ok then return nil, err end
    if n > 0 then
        ok, err = client:send(ffi.string(floats, n * 4))
        if not ok then return nil, err end
    end
    if info and info_len and info_len > 0 then
        ok, err = client:send(ffi.string(info, info_len * 4))
        if not ok then return nil, err end
    end
    return true
end

function Proto.send_log(client, msg)
    return send_msg(client, Proto.MSG_LOG, msg, #msg)
end

function Proto.send_ping(client, t)
    _msgbuf[0] = Proto.MSG_PING
    ffi.cast("float*", _msgbuf + 4)[0] = t
    return send_msg(client, Proto.MSG_PING, ffi.string(_msgbuf, 8), 8)
end

--------------------------------------------------------------------------------
-- Receiving
--------------------------------------------------------------------------------

--- Block until one message arrives.
--  @return tag, payload_string (payload may be an empty string)
function Proto.recv_msg(client)
    local raw, err = client:receive(HDR_SZ)
    if not raw then return nil, err end
    local hdr = ffi.cast("rl_hdr_t*", raw)
    local tag = hdr.tag
    local len = hdr.len
    local payload = ""
    if len > 0 then
        payload, err = client:receive(len)
        if not payload then return nil, err end
    end
    return tag, payload
end

--- Decode the fixed 32-byte control payload (MSG_ACT / MSG_EPISODE / MSG_QUIT).
function Proto.decode_control(payload)
    if payload == nil or #payload < 32 then
        return { msg = Proto.MSG_QUIT }
    end
    local m = ffi.cast("rl_msg_t*", payload)
    return {
        msg         = m.msg,
        action      = m.action,
        boss_index  = m.boss_index,
        seed        = m.seed,
        max_steps   = m.max_steps,
        param0      = m.param0,
        param1      = m.param1,
        param2      = m.param2,
    }
end

return Proto