-- ============================================================================
--  rl_main.lua -- the harness main loop.
--
--  One LÖVE process serves exactly one environment.  Python connects, sends an
--  episode spec, then streams actions while the harness streams observations.
--
--  Two pacing modes:
--    headless : fixed dt, no drawing, no sleeping    -> training
--    realtime : fixed dt + drawing + wall-clock pacing -> watching the agent
--
--  Both speak the identical protocol so `play`, `eval` and `train` share code.
--
--  Wire protocol (see rl_proto.lua / godhome_rl/env/proto.py)
--  ----------------------------------------------------------------------
--    Lua <- MSG_EPISODE   start a new fight with the given spec
--    Lua <- MSG_ACT       apply this action, then report
--    Lua -> MSG_STEP      obs, reward, flags, info
--    Lua -> MSG_LOG       diagnostics
-- ============================================================================

local Config = require("rl.rl_config")
local Net    = require("rl.rl_net")
local Proto  = require("rl.rl_proto")
local Obs    = require("rl.rl_obs")
local Battle = require("rl.rl_battle")

local Main = {}

Main.ACTIONS = 36          -- 9 movement directions x attack x focus

local function write_port_file(path, port)
    if not path then return end
    local f = io.open(path, "w")
    if f then
        f:write(string.format("%d", port))
        f:close()
    end
end

local function do_draw(dt)
    if not love.graphics.isActive() then return end
    love.graphics.origin()
    love.graphics.clear(love.graphics.getBackgroundColor())
    love.draw(dt)
    love.graphics.present()
end

-- Observer mode only needs to know whether the window was closed; the policy
-- supplies every other input, so we deliberately do not dispatch handlers here.
-- `love.event.poll()` is type-checked because LÖVE builds differ in what they
-- return when the queue is empty.
local function pump_events()
    if not love.event then return false end
    love.event.pump()
    local ok, events = pcall(love.event.poll)
    if not ok or type(events) ~= "table" then return false end
    for _, ev in ipairs(events) do
        if ev[1] == "quit" then return true end
    end
    return false
end

local function episode_over(flags)
    return math.floor(flags / Proto.FLAG_EPISODE) % 2 == 1
end

--------------------------------------------------------------------------------
-- Main serve loop
--------------------------------------------------------------------------------

function Main.serve(config)
    -- Listen *before* booting so a crash during asset loading still reaches
    -- Python as a readable error instead of an opaque timeout.
    local server, port = Net.listen(config.port)
    print(string.format("GODHOME_RL_READY port=%d", port))
    io.stdout:flush()
    write_port_file(config.port_file, port)

    local client = Net.accept(server)
    if not client then
        Config.log("trainer never connected")
        server:close()
        return 1
    end
    RL.client = client

    local ok, err = pcall(Battle.boot, config)
    if not ok then
        local dump = (_G.RL and _G.RL.trace_dump) and _G.RL.trace_dump() or "(no trace)"
        Proto.send_log(client, "boot failed: " .. tostring(err))
        Proto.send_log(client, "boot trace:\n" .. dump)
        Net.close(client)
        server:close()
        error(err, 0)
    end

    local dim = Obs.configure(config.grid_w, config.grid_h, config.max_bullets)
    -- Ordered: Python mirrors this order in env/proto.py::Handshake.
    Proto.send_ready(client, {
        1,                      -- proto version
        dim,                    -- observation dimensionality
        Main.ACTIONS,           -- discrete action count
        config.dt,              -- fixed timestep (s)
        config.grid_w,
        config.grid_h,
        config.max_bullets,
        Obs.GLOBAL_SIZE,
        Obs.GRID_CHANNELS,
        Obs.BULLET_FEATURES,
        16,                     -- info vector length
    })

    local BOSSES = config.boss_table or {}

    local running = true
    local frame = 0
    local next_deadline = love.timer.getTime()

    local tag, payload = Proto.recv_msg(client)
    local spec = (tag == Proto.MSG_EPISODE) and Proto.decode_control(payload) or nil

    while running and spec do
        ------------------------------------------------------------ episode --
        local boss_id = spec.boss_id or BOSSES[(spec.boss_index or 0) + 1]
        local ok = Battle.start(boss_id, {
            max_steps  = spec.max_steps,
            seed       = spec.seed,
            difficulty = tonumber(spec.param0) or 1,
            variant    = (spec.param1 or 0) ~= 0 and "ascended" or "attuned",
        })
        if not ok then
            Proto.send_log(client, "episode start failed: " .. tostring(Battle.error))
        end
        Obs.reset_cache()

        ------------------------------------------------------- inner steps --
        -- The harness fast-forwards through menus/dialogue on its own and only
        -- hands Python a frame once the soul is actually in play.
        local advance_reward, flags = Battle.advance_to_action(config)
        local reward = 0
        if not episode_over(flags) then
            -- The fight is live and waiting for the first decision.
            reward = 0
            flags = Proto.FLAG_RESET
        end
        local info, info_len = Battle.info_buf, 16
        local first = true
        local first_frame = true

        while true do
            if not first then
                local atag, apayload = Proto.recv_msg(client)
                if not atag or atag == Proto.MSG_QUIT then
                    running = false
                    break
                end
                local act = Proto.decode_control(apayload).action
                if act < 0 then act = 0 end
                reward, flags, info, info_len = Battle.tick(act, config)
                frame = frame + 1
            end
            first = false

            local buf = Obs.build(Battle.obs_extra())
            -- Grazes are counted while the observation is built.
            if not first_frame then
                reward = reward + Battle.graze_bonus(config)
            end
            first_frame = false
            if not Proto.send_step(client, buf, dim, reward, flags, info, info_len) then
                running = false
                break
            end

            if episode_over(flags) then break end

            if config.render and (frame % config.render_every == 0) then
                if pump_events() then running = false break end
                do_draw(config.dt)
                next_deadline = next_deadline + (config.dt * config.render_every)
                local slack = next_deadline - love.timer.getTime()
                if slack > 0 then love.timer.sleep(slack) end
            end
        end
        if not running then break end

        -------------------------------------------------------- next episode
        tag, payload = Proto.recv_msg(client)
        if not tag or tag == Proto.MSG_QUIT then break end
        spec = Proto.decode_control(payload)
    end

    Net.close(client)
    server:close()
    return 0
end

function Main.run_headless()
    return Main.serve(Config)
end

function Main.run_realtime()
    -- Observer mode: identical protocol, but the loop renders and paces itself
    -- to wall-clock time so a human can watch the policy play.
    return Main.serve(Config)
end

return Main