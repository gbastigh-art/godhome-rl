-- ============================================================================
--  rl_boot.lua -- GodhomeRL harness bootstrap.
--
--  Required from main.lua right after the engine's own requires, i.e. once
--  every Kristal global exists and Kristal's own love.run is defined.  At this
--  point we can safely replace love.run with our own headless scheduler.
--
--  Everything is gated on the GODHOME_RL environment variable, so a patched
--  build with the variable unset behaves exactly like vanilla Godhome.
--
--  LÖVE is unforgiving about startup errors (it can show an error screen and
--  carry on), so the whole bootstrap runs inside a pcall that writes a
--  breadcrumb file when GODHOME_RL_DEBUG points at one.
-- ============================================================================

local RL = {}

RL.env = os.getenv("GODHOME_RL") or ""
RL.debug_path = os.getenv("GODHOME_RL_DEBUG")
RL._trace_lines = {}

local function trace(msg)
    msg = tostring(msg)
    if #RL._trace_lines < 4000 then
        RL._trace_lines[#RL._trace_lines + 1] = msg
    end
    if not RL.debug_path then return end
    local f = io.open(RL.debug_path, "a")
    if f then
        f:write(msg, "\n")
        f:close()
    end
end
RL.trace = trace
RL.debug = RL.debug_path ~= nil
RL.state_counts = {}
RL.state_frames = 0

function RL.trace_dump()
    return table.concat(RL._trace_lines, "\n")
end

if RL.env == "" then
    return
end

trace("boot: GODHOME_RL set (" .. #RL.env .. " bytes)")

local function main()
    RL.config = require("rl.rl_config")
    RL.proto   = require("rl.rl_proto")
    RL.obs     = require("rl.rl_obs")
    RL.battle  = require("rl.rl_battle")
    RL.net     = require("rl.rl_net")
    RL.main    = require("rl.rl_main")

    RL.config.parse(RL.env)
    trace("config parsed: dt=" .. tostring(RL.config.dt)
          .. " bosses=" .. tostring(RL.config.bosses))

-- LuaJIT escape hatch.
--
-- The harness matches the JIT settings the game ships with, which measured the
-- same throughput as the interpreter here.  We did see impossible LuaJIT errors
-- during development ("attempt to index global '('" inside the engine), but the
-- root cause turned out to be an observation-buffer overflow rather than the
-- compiler.  `GODHOME_RL_JIT=off` disables the compiler entirely if you ever
-- hit instability on an unusual machine, and `=flush` drops the trace cache
-- between fights as a middle ground.
local jit_ok, jit = pcall(require, "jit")
RL.jit = jit
RL.jit_mode = os.getenv("GODHOME_RL_JIT") or "on"
if jit_ok then
    if RL.jit_mode == "off" then
        jit.off(true, true)
    end
    trace("LuaJIT mode: " .. RL.jit_mode)
end

    -- LuaSocket is compiled into LÖVE >= 11.1; without it there is no way to
    -- talk to Python, so fail loudly instead of hanging on listen().
    if not pcall(require, "socket") then
        error("GodhomeRL: LuaSocket is unavailable in this LÖVE build.")
    end
    trace("socket ok")

    ---------------------------------------------------------------- loop --
    function love.run()
        love.timer.step()
        trace("love.run entered")
        local ok, err = pcall(function()
            if RL.config.render then
                return RL.main.run_realtime()
            end
            return RL.main.run_headless()
        end)
        if not ok then
            trace("LOOP ERROR: " .. tostring(err))
            -- Best effort: tell the trainer *why* before the socket dies, so a
            -- crashed harness shows up as a readable message instead of a
            -- "connection reset" in the middle of a training run.
            if RL.client then
                pcall(function()
                    RL.proto.send_log(RL.client, "harness error: " .. tostring(err))
                end)
            end
            error(err, 0)
        end
        return 0
    end

    _G.RL = RL
    trace("bootstrap complete")
    return RL
end

local ok, err = pcall(main)
if not ok then
    trace("BOOTSTRAP ERROR: " .. tostring(err))
    error(err, 0)
end

return _G.RL