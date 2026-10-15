-- ============================================================================
--  rl_battle.lua -- boots the game, owns the episode lifecycle, and shapes
--  rewards.
--
--  Episode lifecycle
--  -----------------
--    boot()            load assets once, drop straight into the Godhome mod
--    start(boss_id)    wipe the stage and launch `boss_id` with fixed rules
--    tick(action)      apply action -> one fixed-dt engine step -> reward
--
--  Everything here is written so that starting a new episode costs a handful of
--  milliseconds (no asset reload, no menu, no map), which is what makes a
--  4-core laptop a usable RL box.
-- ============================================================================

local Battle = {}

local ffi = require("ffi")
local Proto = require("rl.rl_proto")
local Obs = require("rl.rl_obs")

-- Breadcrumb logging.  Resolved lazily because rl_boot installs the sink after
-- this module is required.
local function trace(msg)
    if _G.RL and _G.RL.trace then _G.RL.trace(msg) end
end


--------------------------------------------------------------------------------
-- Boot
--------------------------------------------------------------------------------

local SETTINGS = [[
{
  "fps": 30,
  "vSync": false,
  "skipIntro": true,
  "fullscreen": false,
  "masterVolume": 0,
  "musicVolume": 0,
  "soundVolume": 0,
  "windowScale": 1,
  "masterVolume": 0,
  "systemCursor": false,
  "alwaysShowCursor": false,
  "shakeReduction": 1,
  "targetSystem": true,
  "autoRun": false,
  "debug": false,
  "simpleShadow": true,
  "reduceEffects": 1,
  "favorites": []
}
]]

--- Load Kristal + the Godhome mod once and enter a neutral Game state.
--  Must be called from inside `love.run`, after love.timer has been stepped.
local function clear_stage(stage)
    if not stage then return end
    for _, child in ipairs(stage.children) do
        pcall(function() stage:removeFromStage(child) end)
    end
    stage.children = {}
end

--- A World instance that is completely inert.
--
--  Godhome boss fights happen on the battle stage; the overworld is dead
--  weight.  Kristal still constructs a World and still pokes at its `music`,
--  `player` and `followers`, so we keep a real World object (cheap: an empty
--  Map, a Camera and a StateManager) but neutralise update/draw so we never
--  walk an overworld that was never loaded.
local function make_inert_world()
    local world = World()
    world.map = Map(world)
    world.player = nil
    world.followers = {}
    world.update = function() end
    world.draw = function() end
    return world
end


function Battle.boot(config)
    local T = _G.RL and _G.RL.trace or function() end
    local t0 = os.clock()
    local function T2(msg)
        T(string.format("%-46s %6.1fs cpu  tick %d", msg, os.clock() - t0, TICK))
    end
    TICK = 0
    love.filesystem.write("settings.json", SETTINGS)
    love.filesystem.remove("keybinds.json")

    local args = love.arg.parseGameArguments(arg)
    love.load(args, arg)
    T2("love.load returned")
    T("skipIntro=" .. tostring(Kristal.Config["skipIntro"])
      .. " fps=" .. tostring(Kristal.Config["fps"]))

    -- Pump the engine's own loading state until all assets have streamed in.
    local guard = 0
    local state = nil
    local last_report = 0
    while guard < 2000000 do
        guard = guard + 1
        TICK = guard
        Input.clear()
        love.update(1 / 30)
        local cur = Gamestate.current()
        if cur ~= state then
            state = cur
            T2("gamestate changed")
        end
        if guard - last_report > 20000 then
            last_report = guard
            T2("... waiting, loader.waiting=" .. tostring(Kristal.Loader.waiting)
               .. " load_complete=" .. tostring(cur and cur.load_complete))
        end
        if cur == Kristal.States["Loading"] then
            if cur.load_complete and cur.key_check and
               (cur.animation_done or Kristal.Config["skipIntro"]) then
                break
            end
        elseif cur == Kristal.States["Menu"] or cur == Kristal.States["Game"] then
            break
        end
    end
    T2("base assets loaded")

    -- Kill the Menu if we ended up there.
    if Gamestate.current() ~= Kristal.States["Game"] then
        -- Load the mod and go straight into the Game state with no map.
        Mod = nil
        local ready = false
        local function after()
            if ready then return end
            ready = true
            T2("mod assets ready")
            Mod.info.map = nil            -- skip overworld map loading
            Mod.info.encounter = nil      -- we start encounters manually
            Kristal.preInitMod("godhome")
            Gamestate.switch(Kristal.States["Game"], nil, nil)
            T2("switched to Game")
        end
        Kristal.loadMod("godhome", nil, nil, function()
            T2("loadMod transition done")
            Kristal.loadModAssets("godhome", "all", "", after)
        end)
        local g2 = 0
        while Gamestate.current() ~= Kristal.States["Game"] and g2 < 2000000 do
            g2 = g2 + 1
            TICK = TICK + 1
            Input.clear()
            love.update(1 / 30)
        end
        T2("game state reached")
    end

    -- Silence + neutralise any residual Game state (the hub may have loaded).
    clear_stage(Game.stage)
    Game.battle = nil
    Game.shop = nil
    Game.world.map = nil
    Game.state = "OVERWORLD"
    Game:setFlag("in_pantheon", false)
    Game:setFlag("bindings", {})
    Game:setTension(0)
    Game:setMaxTension(100)

    -- Fixed-capacity info buffer shared by every step (avoids GC churn).
    Battle.info_buf = ffi.new("float[16]")
end

--------------------------------------------------------------------------------
-- Episode control
--------------------------------------------------------------------------------

local ep = {
    boss          = "p1/falseknight",
    frames        = 0,
    max_frames    = 1800,
    frames_since_hit = 999,
    hit_count     = 0,
    graze_count   = 0,
    boss_hp_start = 0,
    player_hp     = 0,
    player_hp_min = 0,
    max_hp        = 90,
    damage_dealt  = 0,
    damage_taken  = 0,
    swings        = 0,
    hits_to_boss  = 0,
    prev_soul_x   = nil,
    prev_soul_y   = nil,
    last_action   = -1,
    running       = false,
    finished      = false,
    outcome       = 0,      -- 1 = victory, 0 = running, -1 = defeat
    frozen_until  = 0,
    idle_steps    = 0,
    spells_seen   = {},
    enc           = nil,
}

Battle.ep = ep

--- The Knight's party member data, or nil outside a battle.
local function party_chara()
    local b = Game.battle
    if b and b.party and b.party[1] then
        return b.party[1].chara
    end
    return nil
end

--- Wipe whatever is on the stage and start a fresh boss fight.
function Battle.start(boss_id, opts)
    opts = opts or {}
    local mod = Mod or {}
    mod.info = mod.info or {}

    -- Drop stale JIT traces between fights (see rl_boot for why).
    if _G.RL and _G.RL.jit and _G.RL.jit_mode ~= "off" then
        pcall(_G.RL.jit.flush)
    end

    ep.boss       = boss_id or ep.boss
    ep.frames     = 0
    ep.agent_frames = 0
    if _G.RL then _G.RL.state_counts = {} _G.RL.state_frames = 0 end
    ep.frames_since_hit = 999
    ep.hit_count  = 0
    ep.graze_count= 0
    ep.damage_dealt = 0
    ep.damage_taken = 0
    ep.prev_soul_x = nil
    ep.prev_soul_y = nil
    ep.last_action = -1
    ep.finished   = false
    ep.outcome    = 0
    ep.spells_seen = {}
    ep.max_frames = opts.max_steps or ep.max_frames
    ep.variant    = opts.variant or "attuned"
    ep.seed       = opts.seed

    -- Clean slate ------------------------------------------------------- //
    if Game.battle then
        pcall(function() Game.battle:remove() end)
        Game.battle = nil
    end
    if Game.shop then pcall(function() Game.shop:remove() end) end
    if Game.gameover then pcall(function() Game.gameover:remove() end) end
    Game.shop = nil
    Game.gameover = nil
    clear_stage(Game.stage)
    Game.stage = Stage()
    Game.fader = Fader()
    Game.fader.layer = 1000
    Game.stage:addChild(Game.fader)
    Game.world = make_inert_world()
    Game.stage:addChild(Game.world)
    Game.state = "OVERWORLD"

    -- Rules / flags ------------------------------------------------------ //
    Game:setFlag("in_pantheon", false)
    Game:setFlag("bindings", opts.bindings or {})
    Game:setFlag("hitless", false)
    Game:setFlag("taught_healing", true)
    Game:setTension(0)
    Game:setMaxTension(100)
    Game.money = 0
    Game.xp = 0

    -- Party ------------------------------------------------------------- //
    local chara = Game.party[1]
    chara.health = chara.stats.health
    ep.max_hp = chara.stats.health
    ep.player_hp = chara.health

    -- Difficulty -------------------------------------------------------- //
    local diff = opts.difficulty or 1
    Game:setFlag("rl_difficulty", diff)
    Game:setFlag("rl_variant", ep.variant)

    -- Launch ------------------------------------------------------------ //
    mod.info.encounter = ep.boss
    local ok, err = pcall(function()
        Game:encounter(ep.boss, false)
    end)
    if not ok then
        if _G.RL and _G.RL.debug then
            local names = {}
            for _, child in ipairs(Game.stage.children) do
                names[#names + 1] = tostring(child.name) .. "/" .. type(child.update)
            end
            trace("stage children after failed encounter: " .. table.concat(names, ", "))
        end
        trace("encounter failed: " .. tostring(err))
    end
    if not ok then
        Battle.error = tostring(err)
        ep.finished = true
        ep.outcome = -1
        return false
    end

    ep.enc = Game.battle.encounter
    if ep.enc and opts.difficulty then
        ep.enc.difficulty = opts.difficulty
    end

    local boss = ep.enc.bosses and ep.enc.bosses[1]
    ep.boss_hp_start = (boss and (boss.max_health or 0)) or 0
    ep.boss_total = 0
    if boss then
        ep.boss_total = boss.max_health or 0
        if boss.armor_health then ep.boss_total = ep.boss_total + (boss.armor_health * boss.phase or boss.armor_health) end
    end

    Battle.error = nil
    return true
end

--------------------------------------------------------------------------------
-- Input synthesis
--------------------------------------------------------------------------------

-- 9-way movement + the two Godhome verbs.
local DIRS = {
    [0] = nil,
    [1] = {"left"},            [2] = {"right"},
    [3] = {"up"},              [4] = {"down"},
    [5] = {"left", "up"},      [6] = {"right", "up"},
    [7] = {"left", "down"},    [8] = {"right", "down"},
}

-- Action decoding, precomputed once.  Using a lookup table instead of
-- math.floor(action / 9) keeps the hot step free of divisions *and* sidesteps a
-- LuaJIT miscompile we hit when dividing inside the per-frame path.
local A_MOVE, A_ATTACK, A_FOCUS = {}, {}, {}
for _a = 0, 35 do
    local m = _a % 9
    local b = _a - m            -- == 9 * floor(a / 9), exactly, for a < 36
    A_MOVE[_a] = m
    A_ATTACK[_a] = (b >= 9)
    A_FOCUS[_a] = (b >= 18)
end

--- Decode an action index into keyboard state and inject it into Kristal.
--  Menu navigation is driven through the real `love.keypressed` dispatch so the
--  exact same code paths run as in a human game; soul movement / nail / focus
--  are read straight off Kristal's Input tables by the Soul class.
local function inject(action, autopilot)
    local attack, focus = false, false
    -- `clear_down` MUST be true.  Kristal's `Input.clear()` only resets
    -- key_pressed/key_released; key_down survives unless asked.  Leaving a
    -- direction key held makes it sticky, and because Soul:doMovement sums
    -- `Input.down("left") .. Input.down("right")` a stale pair cancels out and
    -- the soul is pinned to the spot for the rest of the fight.
    Input.clear(nil, true)
    if autopilot then
        if _G.RL and _G.RL.debug then
            local b = Game.battle
            local st = b and b.state or "NONE"
            _G.RL.state_counts[st] = (_G.RL.state_counts[st] or 0) + 1
            _G.RL.state_frames = (_G.RL.state_frames or 0) + 1
            if _G.RL.state_frames % 200 == 0 then
                local parts = {}
                for k, v in pairs(_G.RL.state_counts) do parts[#parts + 1] = k .. "=" .. v end
                table.sort(parts)
                trace("states: " .. table.concat(parts, " "))
            end
        end
        love.keypressed("z")
        Input.key_down.z = nil

        -- Skip the purely cosmetic parts of the turn loop.  These frames are
        -- never shown to the policy, so paying for them only wastes simulator
        -- time: roughly a third of a boss fight is dialogue boxes and the
        -- arena spawn animation.  Only *timers* are touched -- no state is
        -- skipped, so the fight plays out exactly as it would for a human who
        -- mashes confirm.
        local b = Game.battle
        if b then
            if b.state == "ENEMYDIALOGUE" then
                b.textbox_timer = 0        -- advance textboxes immediately
            elseif b.state == "DEFENDINGBEGIN" then
                b.defending_begin_timer = 15
            elseif b.state == "INTRO" then
                b.intro_timer = 15
            end
        end
    else
        attack = A_ATTACK[action]
        focus = A_FOCUS[action]
        local dirs = DIRS[A_MOVE[action]]
        if dirs then
            for i = 1, #dirs do Input.key_down[dirs[i]] = true end
        end
        if attack then Input.key_pressed.z = true end
        if focus then Input.key_down.x = true end
    end
    return attack, focus
end

--------------------------------------------------------------------------------
-- State helpers
--------------------------------------------------------------------------------

local function in_defend_phase()
    local b = Game.battle
    return b ~= nil and b.state == "DEFENDING" and b.soul ~= nil
end

local STATE_CODE = {
    gameover = 1, no_battle = 2, dream_end = 3, transitionout = 4,
}

local function is_over()
    local b = Game.battle
    local st = b and b.state or "NONE"
    if Game.state == "GAMEOVER" then
        ep.over_reason, ep.over_state = "gameover", st
        return "death"
    end
    if not b then
        ep.over_reason, ep.over_state = "no_battle", st
        return ep.finished and (ep.outcome == 1 and "win" or "death")
    end
    if b.state == "DREAM_END" then
        ep.over_reason, ep.over_state = "dream_end", st
        return "win"
    end
    if b.state == "TRANSITIONOUT" and ep.enc and #ep.enc.bosses == 0 then
        ep.over_reason, ep.over_state = "transitionout", st
        return "win"
    end
    ep.over_reason, ep.over_state = nil, st
    return nil
end

--- Combined boss health as a fraction of the fight's total pool.
local function current_boss_frac()
    local enc = ep.enc
    if not enc then return 1, 0 end
    local total, hp = 0, 0
    for _, b in ipairs(enc.bosses or {}) do
        local mh = b.max_health or 0
        if mh > 0 then
            total = total + mh
            hp = hp + math.max(0, b.health)
        end
        -- Armour counts as its own pseudo-health pool so that the agent sees
        -- "break the armour, then the boss" as one monotone progress bar.
        if b.armor_health and b.armor_health > 0 then
            total = total + 300
            hp = hp + math.max(0, math.min(1, b.armor_health / 300) * 300)
        end
    end
    if total <= 0 then return 1, 0 end
    return hp / total, total
end

local function party_hp()
    local c = party_chara()
    return c and c.health or ep.player_hp
end

--------------------------------------------------------------------------------
-- One engine frame + its reward
--------------------------------------------------------------------------------

--  @return reward
local function engine_frame(action, config, autopilot)
    local pre_hp = party_hp()
    local frac_before = current_boss_frac()
    local _, focus = inject(action, autopilot)

    -- Snapshot the soul *before* the engine moves it.  Obs.build derives the
    -- velocity channels from `extra.prev_soul_x`, so this has to be the
    -- position from the previous frame; sampling it afterwards would compare
    -- the soul against itself and report zero velocity forever.
    local soul = Game.battle and Game.battle.soul
    if soul then
        ep.prev_soul_x, ep.prev_soul_y = soul.x, soul.y
    end

    love.update(config.dt)

    local reward = 0
    local hp = party_hp()
    soul = Game.battle and Game.battle.soul

    -- damage taken
    local d_taken = pre_hp - hp
    if d_taken > 0 then
        ep.hit_count = ep.hit_count + 1
        ep.frames_since_hit = 0
        ep.damage_taken = ep.damage_taken + d_taken
        reward = reward - (config.r_damage * d_taken / ep.max_hp)
    else
        ep.frames_since_hit = ep.frames_since_hit + 1
    end

    -- damage dealt to the boss pool
    local frac_after = current_boss_frac()
    local d_dealt = frac_before - frac_after
    if d_dealt > 0 then
        ep.damage_dealt = ep.damage_dealt + d_dealt
        reward = reward + (config.r_damage_dealt * d_dealt)
    end

    -- survival bonus, only while a bullet pattern is actually running
    if in_defend_phase() then
        reward = reward + config.r_alive
        if focus then reward = reward - config.r_focus end
    end

    return reward, soul, hp
end

--------------------------------------------------------------------------------
-- Public stepping API
--------------------------------------------------------------------------------

--: Upper bound on auto-piloted frames in a single network step.  A turn should
--: never need more than a couple of hundred, but a pathological boss must not
--: be allowed to stall the trainer.
local AUTOPILOT_BUDGET = 2000

--- True when the battle object exists but has not been initialised.
--
--  A handful of Godhome encounters leave ``Game.battle`` in this state after a
--  wave ends (it is the value ``Battle:init`` sets, before ``postInit`` runs).
--  Nothing progresses, so the harness treats it as the end of the fight rather
--  than burning the frame budget on it.
local function battle_is_idle()
    local b = Game.battle
    return b ~= nil and b.state == "NONE"
end

--- One network step.
--
--  The agent's action is consumed on the next *bullet-hell* frame.  Any number
--  of intervening menu / dialogue / wave-transition frames are auto-confirmed
--  by the harness first, so every transition Python sees is a frame where the
--  decision actually changes something.
function Battle.tick(action, config)
    local reward = 0
    local guard = 0
    while true do
        if in_defend_phase() or ep.finished then
            if not ep.finished then
                ep.frames = ep.frames + 1
                ep.agent_frames = ep.agent_frames + 1
                ep.last_action = action
                reward = reward + (engine_frame(action, config, false))
            end
            break
        end
        guard = guard + 1
        ep.frames = ep.frames + 1
        reward = reward + (engine_frame(18, config, true))
        if is_over() or ep.frames >= ep.max_frames then break end
        if battle_is_idle() then
            ep.finished = true
            ep.outcome = (ep.enc and #ep.enc.bosses == 0) and 1 or 0
            break
        end
        if guard > AUTOPILOT_BUDGET then
            ep.finished = true
            ep.outcome = 0
            break
        end
    end
    return Battle.finalise(reward, config)
end

--- Fast-forward until the soul is in play (or the fight ends) and report the
--  first decision point of the episode.
function Battle.advance_to_action(config)
    local reward = 0
    local guard = 0
    while not ep.finished and not in_defend_phase() do
        guard = guard + 1
        ep.frames = ep.frames + 1
        reward = reward + (engine_frame(18, config, true))
        if is_over() or ep.frames >= ep.max_frames then break end
        if battle_is_idle() then
            ep.finished = true
            ep.outcome = (ep.enc and #ep.enc.bosses == 0) and 1 or 0
            break
        end
        if guard > AUTOPILOT_BUDGET then
            ep.finished = true
            ep.outcome = 0
            break
        end
    end
    return reward, 0
end

--- Shared tail: termination checks, info vector and the step flags.
function Battle.finalise(reward, config)
    local flags = 0
    local outcome = is_over()

    if outcome == "win" then
        if not ep.finished then reward = reward + config.r_win end
        ep.finished = true
        ep.outcome = 1
        flags = Proto.FLAG_TERMINAL + Proto.FLAG_EPISODE
    elseif outcome == "death" then
        if not ep.finished then reward = reward + config.r_death end
        ep.finished = true
        ep.outcome = -1
        flags = Proto.FLAG_TERMINAL + Proto.FLAG_EPISODE
    elseif ep.frames >= ep.max_frames then
        reward = reward + config.r_timeout
        ep.finished = true
        ep.outcome = 0
        flags = Proto.FLAG_TERMINAL + Proto.FLAG_TRUNCATED + Proto.FLAG_EPISODE
    end

    local frac = current_boss_frac()
    local hp = party_hp()
    local info = Battle.info_buf
    info[0]  = ep.frames
    info[1]  = frac
    info[2]  = hp
    info[3]  = ep.max_hp
    info[4]  = ep.damage_dealt
    info[5]  = ep.damage_taken
    info[6]  = ep.graze_count
    info[7]  = ep.hit_count
    info[8]  = (ep.outcome == 1 and 1) or (ep.outcome == -1 and -1) or 0
    info[9]  = ep.boss_total
    info[10] = Obs.graze_total
    info[11] = ep.over_reason and (STATE_CODE[ep.over_reason] or 0) or 0
    info[14] = Obs.debug_nbullets
    info[15] = ep.agent_frames
    -- `ep.prev_soul_*` is deliberately not touched here: engine_frame samples
    -- it *before* love.update so that Obs.build sees a real frame-to-frame
    -- delta.  Writing it now would compare the soul against its own position.

    return reward, flags, info, 16
end

--- Grazes are tallied while the observation is built (the engine flips
--  ``bullet.grazed``, the cheapest exact signal available), so the reward is
--  collected *after* ``Obs.build``.  Returns the bonus to add to this step.
function Battle.graze_bonus(config)
    local new_grazes = Obs.take_graze_count()
    if new_grazes <= 0 then return 0 end
    ep.graze_count = ep.graze_count + new_grazes
    Obs.graze_total = Obs.graze_total + new_grazes
    if not in_defend_phase() then return 0 end
    return config.r_graze * new_grazes
end

function Battle.obs_extra()
    return {
        prev_soul_x        = ep.prev_soul_x,
        prev_soul_y        = ep.prev_soul_y,
        frames             = ep.frames,
        max_steps          = ep.max_frames,
        frames_since_hit   = ep.frames_since_hit,
        hit_count          = ep.hit_count,
        graze_count        = ep.graze_count,
        last_action        = ep.last_action,
    }
end

return Battle
