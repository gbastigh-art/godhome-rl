-- ============================================================================
--  rl_obs.lua -- turns a live Battle into a flat float32 vector.
--
--  The vector is the concatenation of three blocks:
--
--    [ GLOBAL | fixed 48 floats ]
--    [ GRID   | 2 * grid_w * grid_h floats ]   spatial density + threat field
--    [ BULLETS| max_bullets * 10 floats ]     nearest-N bullet descriptors
--
--  The layout constants below MUST stay in sync with
--  `godhome_rl/env/layout.py`, which asserts the same sizes at handshake time.
-- ============================================================================

local ffi = require("ffi")
local Obs = {}

Obs.GLOBAL_SIZE = 50
Obs.BULLET_FEATURES = 10
Obs.GRID_CHANNELS = 2

--: Extra floats allocated past the end of the vector.  Every write is bounds
--: checked anyway, but slack keeps an off-by-one from ever becoming an
--: out-of-bounds FFI write (which corrupts the Lua heap rather than raising).
local OBS_SLACK = 64

local GLOBAL_SIZE     = Obs.GLOBAL_SIZE
local BULLET_FEATURES = Obs.BULLET_FEATURES

-- Scaling constants (pixels)
local REL_SCALE_X = 160.0
local REL_SCALE_Y = 120.0
local VEL_SCALE   = 6.0
local RAD_SCALE   = 20.0
local LOOKAHEAD   = 9      -- frames of linear extrapolation for threat/TTC

local SQRT, COS, SIN, MIN, MAX, ABS = math.sqrt, math.cos, math.sin, math.min, math.max, math.abs

--- Radius (px) of the soul's graze ring, mirroring ``Soul.graze_collider``.
local GRAZE_RADIUS = 25

--------------------------------------------------------------------------------
-- Per-process state
--------------------------------------------------------------------------------

local state = {
    grid_w   = 20,
    grid_h   = 15,
    max_bullets = 24,
    buf      = nil,          -- ffi float array
    dim      = 0,
    velcache = {},           -- [bullet] = {x, y, frame, vx, vy}
    frame    = 0,
    sweep_in = 0,
    graze_new = 0,
    graze_engine = 0,
    tp_bullets = 0,
    near_bullets = 0,
    -- scratch bullet list
    bl = {},                 -- array of bullet records
    bn = 0,
    -- selection
    sel_d = {},
    sel_i = {},
}

--- Number of grazes credited since the last call.
function Obs.take_graze_count()
    local n = state.graze_new
    state.graze_new = 0
    return n
end

--- Cumulative counters, surfaced through the info vector for telemetry.
Obs.graze_total = 0
Obs.graze_engine_total = 0
Obs.debug_tp = 0
Obs.debug_near = 0
Obs.debug_nbullets = 0

function Obs.reset_counters()
    Obs.graze_total = 0
    Obs.graze_engine_total = 0
    state.graze_new = 0
    state.graze_engine = 0
end

function Obs.configure(grid_w, grid_h, max_bullets)
    state.grid_w   = grid_w or 20
    state.grid_h   = grid_h or 15
    state.max_bullets = max_bullets or 24
    state.dim = GLOBAL_SIZE
                + Obs.GRID_CHANNELS * state.grid_w * state.grid_h
                + state.max_bullets * BULLET_FEATURES
    state.capacity = state.dim + OBS_SLACK
    state.buf = ffi.new("float[?]", state.capacity)
    return state.dim
end

function Obs.dim() return state.dim or Obs.configure() end
function Obs.buffer() return state.buf end
function Obs.reset_cache()
    state.velcache = {}
    state.frame = 0
end

--------------------------------------------------------------------------------
-- Helpers
--------------------------------------------------------------------------------

local function collider_radius(c)
    if not c then return 0 end
    if c.radius then return c.radius end
    if c.x2 then
        local dx, dy = c.x2 - c.x, c.y2 - c.y
        return 0.5 * SQRT(dx * dx + dy * dy)
    end
    if c.points then
        local best = 0
        for _, p in ipairs(c.points) do
            local d = SQRT(p[1] * p[1] + p[2] * p[2])
            if d > best then best = d end
        end
        return best
    end
    if c.width and c.height then
        return 0.5 * SQRT(c.width * c.width + c.height * c.height)
    end
    return 0
end

local function physics_velocity(obj)
    local ph = obj.physics
    if not ph then return 0, 0 end
    local vx, vy = 0, 0
    local dir = ph.direction or 0
    if ph.match_rotation then dir = obj.rotation or 0 end
    if ph.speed and ph.speed ~= 0 then
        vx = vx + COS(dir) * ph.speed
        vy = vy + SIN(dir) * ph.speed
    end
    if ph.gravity and ph.gravity ~= 0 then
        local gd = ph.gravity_direction or 0
        vx = vx + COS(gd) * ph.gravity
        vy = vy + SIN(gd) * ph.gravity
    end
    if ph.speed_x and ph.speed_x ~= 0 then vx = vx + ph.speed_x end
    if ph.speed_y and ph.speed_y ~= 0 then vy = vy + ph.speed_y end
    return vx, vy
end

--------------------------------------------------------------------------------
-- Bullet collection
--------------------------------------------------------------------------------

local function collect_bullets()
    local bl, bn = state.bl, 0
    bn = 0
    state.tp_bullets = 0
    state.near_bullets = 0
    local soul = Game.battle and Game.battle.soul
    local sx = soul and soul.x or SCREEN_WIDTH / 2
    local sy = soul and soul.y or SCREEN_HEIGHT / 2
    local frame = state.frame
    local cache = state.velcache

    for _, bullet in ipairs(Game.stage:getObjects(Bullet)) do
        local c = bullet.collider
        local cx, cy, r
        if c then
            cx = bullet.x + c.x
            cy = bullet.y + c.y
            r = collider_radius(c)
        else
            cx, cy, r = bullet.x, bullet.y, 0
        end

        -- exact velocity when we saw this object last frame, physics otherwise
        local vx, vy
        local e = cache[bullet]
        if e and e[3] == frame - 1 then
            vx, vy = cx - e[1], cy - e[2]
        else
            vx, vy = physics_velocity(bullet)
        end
        cache[bullet] = {cx, cy, frame}

        local dangerous = (c == nil) or (c.collidable ~= false)
        local dmg = bullet:getDamage()

        -- Graze bookkeeping.
        --
        -- The engine flips `bullet.grazed` the first time the soul's graze
        -- circle touches it, which is the authoritative signal; we count it.
        -- Some Godhome bullets opt out of tension entirely, so we *also* track
        -- a geometric proxy (a damaging bullet whose hull comes inside the
        -- graze ring) -- that is what the reward actually uses, because it
        -- rewards the behaviour we want rather than a per-bullet flag.
        if bullet.grazed and not bullet._rl_g then
            bullet._rl_g = true
            state.graze_engine = state.graze_engine + 1
        end
        local grazeable = (bullet.tp and bullet.tp ~= 0) and 1 or 0
        if grazeable == 1 then state.tp_bullets = state.tp_bullets + 1 end
        local ddx, ddy = cx - sx, cy - sy
        local reach = r + GRAZE_RADIUS
        if (ddx * ddx + ddy * ddy) < reach * reach then
            state.near_bullets = state.near_bullets + 1
            if grazeable == 1 and dangerous and dmg and dmg > 0 and not bullet._rl_gn then
                bullet._rl_gn = true
                state.graze_new = state.graze_new + 1
            end
        end

        bn = bn + 1
        local rec = bl[bn]
        if not rec then
            rec = {0,0,0,0,0,0,0,0,0,0,0,0}
            bl[bn] = rec
        end
        rec[1] = cx
        rec[2] = cy
        rec[3] = vx
        rec[4] = vy
        rec[5] = r
        rec[6] = (dangerous and dmg and dmg > 0) and 1 or 0
        rec[7] = (dangerous and dmg == 0) and 1 or 0
        rec[8] = (bullet.tp and bullet.tp ~= 0) and 1 or 0
        rec[9] = (cx - sx) * (cx - sx) + (cy - sy) * (cy - sy)
        rec[10] = bullet
        rec[11] = bullet.grazed and 1 or 0
    end
    state.bn = bn
end

local function sweep_cache()
    local cache = state.velcache
    local frame = state.frame
    for k, e in pairs(cache) do
        if e[3] ~= frame then
            cache[k] = nil
        end
    end
end

--------------------------------------------------------------------------------
-- Nearest-N selection (partial selection sort; N is small)
--------------------------------------------------------------------------------

local function select_nearest()
    local bn = state.bn
    local n  = state.max_bullets
    local bl = state.bl
    local sd, si = state.sel_d, state.sel_i
    local taken = 0
    for i = 1, n do sd[i] = nil; si[i] = nil end

    for _ = 1, math.min(n, bn) do
        local best, bestd
        for i = 1, bn do
            local d = bl[i][9]
            if d < 1e9 then
                if not bestd or d < bestd then
                    bestd = d
                    best = i
                end
            end
        end
        if not best then break end
        taken = taken + 1
        sd[taken] = bestd
        si[taken] = best
        bl[best][9] = 1e9        -- mark consumed
    end

    -- restore distances for the next frame
    for i = 1, taken do
        bl[si[i]][9] = sd[i]
    end
    state.sel_n = taken
end

--------------------------------------------------------------------------------
-- Main entry point
--------------------------------------------------------------------------------

--- Build the observation into the internal buffer and return the buffer.
function Obs.build(extra)
    local buf = state.buf
    local cap = state.capacity
    local n = 0
    local overflow = 0
    -- Bounds-checked writer: an observation-shape mistake must degrade into a
    -- truncated vector, never into an out-of-bounds write.
    local function put(v)
        n = n + 1
        if n <= cap then
            buf[n - 1] = v
        else
            overflow = overflow + 1
        end
    end

    state.frame = state.frame + 1
    if state.sweep_in <= 0 then
        sweep_cache()
        state.sweep_in = 30
    else
        state.sweep_in = state.sweep_in - 1
    end

    local battle = Game.battle
    local soul   = battle and battle.soul or nil
    local sx = soul and soul.x or (SCREEN_WIDTH / 2)
    local sy = soul and soul.y or (SCREEN_HEIGHT / 2)

    collect_bullets()
    Obs.debug_tp = state.tp_bullets
    Obs.debug_near = state.near_bullets
    Obs.debug_nbullets = state.bn
    select_nearest()

    ------------------------------------------------------------------ GLOBAL --
    local e = extra or {}

    -- soul kinematics (positions first, then smoothed per-frame velocity)
    local dx, dy = 0, 0
    if e.prev_soul_x then
        dx = sx - e.prev_soul_x
        dy = sy - e.prev_soul_y
        dx = (dx - (state.last_dx or 0)) * 0.5
        dy = (dy - (state.last_dy or 0)) * 0.5
        state.last_dx, state.last_dy = sx - e.prev_soul_x, sy - e.prev_soul_y
    end
    put(sx / SCREEN_WIDTH)
    put(sy / SCREEN_HEIGHT)
    put(dx / REL_SCALE_X)
    put(dy / REL_SCALE_Y)
    state.prev_x, state.prev_y = sx, sy

    local knight = battle and battle.party and battle.party[1]
    local chara = knight and knight.chara
    local max_hp = (chara and chara.stats and chara.stats.health) or 90
    local hp = (chara and chara.health) or max_hp
    local hp_frac = (hp + 0.5 * max_hp) / (1.5 * max_hp)
    if hp_frac < 0 then hp_frac = 0 elseif hp_frac > 1 then hp_frac = 1 end

    put(hp_frac)
    put(hp / 100)
    put((Game:getTension() or 0) / 100)
    put(soul and soul.inv_timer and (soul.inv_timer / 2) or 0)
    put(soul and soul.can_focus and 1 or 0)
    put(soul and soul.focus_used and (soul.focus_used / 16) or 0)
    put((not soul or soul.cooldown == 0) and 1 or 0)
    put((soul and soul.nail and soul.nail.collidable) and 1 or 0)
    put(soul and soul.knockback and (soul.knockback / 4) or 0)

    -- battle state buckets
    local st = battle and battle.state or "NONE"
    local f_def, f_defb, f_dial, f_act, f_intro, f_end = 0, 0, 0, 0, 0, 0
    if st == "DEFENDING" then f_def = 1
    elseif st == "DEFENDINGBEGIN" then f_defb = 1
    elseif st == "ENEMYDIALOGUE" or st == "DIALOGUEEND" or st == "BATTLETEXT" then f_dial = 1
    elseif st == "ACTIONSELECT" or st == "ACTIONS" or st == "ATTACKING" or st == "ACTIONSDONE"
        or st == "ENEMYSELECT" or st == "PARTYSELECT" or st == "MENUSELECT" or st == "XACTENEMYSELECT" then f_act = 1
    elseif st == "INTRO" or st == "TRANSITION" then f_intro = 1
    elseif st == "VICTORY" or st == "TRANSITIONOUT" or st == "DREAM_END" then f_end = 1
    end
    put(f_def); put(f_defb); put(f_dial); put(f_act); put(f_intro); put(f_end)

    local wlen = (battle and battle.wave_length) or 0
    local wt   = (battle and battle.wave_timer) or 0
    if wlen > 0 then put(wt / wlen) else put(1) end
    put(((wlen - wt) / 5))

    -- bosses (up to 3)
    local boss_i = 0
    if battle then
        for _, enemy in ipairs(battle.enemies or {}) do
            if boss_i >= 3 then break end
            local mh = enemy.max_health or 0
            local hp_ = enemy.health or 0
            put(mh > 0 and (hp_ / mh) or 0)
            local ah = enemy.armor_health
            if ah then
                put(MAX(0, math.min(1, ah / 300)))
            else
                put(-1)
            end
            if boss_i == 0 then
                put((enemy.x - sx) / REL_SCALE_X)
                put((enemy.y - sy) / REL_SCALE_Y)
            else
                put(0); put(0)
            end
            boss_i = boss_i + 1
        end
    end
    for _ = boss_i, 3 do
        put(0); put(-1); put(0); put(0)
    end

    -- arena
    local arena = battle and battle.arena
    if arena then
        put(arena.left / SCREEN_WIDTH)
        put(arena.right / SCREEN_WIDTH)
        put(arena.top / SCREEN_HEIGHT)
        put(arena.bottom / SCREEN_HEIGHT)
        put(arena.width / SCREEN_WIDTH)
        put(arena.height / SCREEN_HEIGHT)
        if arena.area_collider and arena.area_collider.radius then
            put(1)
        else
            put(0)
        end
        put(1)
    else
        put(0); put(1); put(0); put(1); put(0); put(0); put(0); put(0)
    end

    -- episode progress / bookkeeping
    local max_steps = e.max_steps or 1800
    put(MIN(1, (e.frames or 0) / max_steps))
    put(MIN(1, (e.frames_since_hit or 0) / 30))
    put(MIN(1, (e.hit_count or 0) / 10))
    put(MIN(1, (e.graze_count or 0) / 200))

    local spells = battle and battle.encounter and battle.encounter.spells
    put((spells and spells.fury) and 1 or 0)
    put((spells and spells.unn) and 1 or 0)
    put((spells and spells.quickslash) and 1 or 0)
    put((spells and spells.quickfocus) and 1 or 0)
    put((e.last_action or 0) / (BULLET_FEATURES * 10))

    -------------------------------------------------------------------- GRID --
    local gw, gh = state.grid_w, state.grid_h
    local gsize = gw * gh
    local base = n
    for i = 0, gsize * Obs.GRID_CHANNELS - 1 do
        local idx = base + i
        if idx <= cap then buf[idx] = 0 end
    end
    n = n + gsize * Obs.GRID_CHANNELS

    local bullet_radius = 9
    for i = 1, state.sel_n do
        local rec = state.bl[state.sel_i[i]]
        local bx, by, bvx, bvy, br = rec[1], rec[2], rec[3], rec[4], rec[5]
        -- density cell
        local cx = (bx / SCREEN_WIDTH) * gw
        local cy = (by / SCREEN_HEIGHT) * gh
        local gx = math.floor(cx)
        local gy = math.floor(cy)
        if gx >= 0 and gx < gw and gy >= 0 and gy < gh then
            local idx = base + gy * gw + gx
            if idx <= cap then buf[idx] = buf[idx] + 1 end
        end
        -- threat: how close this bullet passes to the soul over the lookahead
        local rx = bx - sx
        local ry = by - sy
        local vv = bvx * bvx + bvy * bvy
        local t = 0
        if vv > 1e-6 then
            t = -(rx * bvx + ry * bvy) / vv
            if t < 0 then t = 0 elseif t > LOOKAHEAD then t = LOOKAHEAD end
        end
        local px = rx + bvx * t
        local py = ry + bvy * t
        local d = SQRT(px * px + py * py) - br - bullet_radius
        local threat = 1 - d / 120
        if threat < 0 then threat = 0 elseif threat > 1 then threat = 1 end
        local px2 = math.floor(((sx + px) / SCREEN_WIDTH) * gw)
        local py2 = math.floor(((sy + py) / SCREEN_HEIGHT) * gh)
        if px2 >= 0 and px2 < gw and py2 >= 0 and py2 < gh then
            local idx = base + gsize + py2 * gw + px2
            if idx <= cap and threat > buf[idx] then buf[idx] = threat end
        end
    end
    -- normalise density
    for i = 0, gsize - 1 do
        local idx = base + i
        if idx <= cap then
            local v = buf[idx] * 0.25
            if v > 1 then v = 1 end
            buf[idx] = v
        end
    end

    ----------------------------------------------------------------- BULLETS --
    local taken = state.sel_n
    local k = n
    local function bput(v)
        if k <= cap then buf[k] = v end
        k = k + 1
    end
    for i = 1, state.max_bullets do
        if i <= taken then
            local rec = state.bl[state.sel_i[i]]
            local bx, by, bvx, bvy, br = rec[1], rec[2], rec[3], rec[4], rec[5]
            local rx = bx - sx
            local ry = by - sy
            bput(rx / REL_SCALE_X)
            bput(ry / REL_SCALE_Y)
            bput(bvx / VEL_SCALE)
            bput(bvy / VEL_SCALE)
            bput(br / RAD_SCALE)
            bput(rec[6])   -- deals damage
            bput(rec[7])   -- harmless
            bput(rec[8])   -- grazeable
            -- time to closest approach (normalised), -1 when never threatening
            local vv = bvx * bvx + bvy * bvy
            local t = 0
            if vv > 1e-6 then
                t = -(rx * bvx + ry * bvy) / vv
                if t < 0 then t = 0 elseif t > LOOKAHEAD then t = LOOKAHEAD end
            end
            bput((t / LOOKAHEAD) * 2 - 1)
            bput(rec[11])  -- already grazed
        else
            for _ = 1, BULLET_FEATURES do bput(0) end
        end
    end

    state.last_count = state.sel_n
    state.overflow = overflow
    return buf, math.min(n, state.dim)
end

return Obs