-- ============================================================================
--  rl_config.lua -- harness configuration.
--
--  Everything arrives through the single GODHOME_RL environment variable using
--  a `key=value;key=value` encoding, so the harness never has to touch
--  Kristal's settings file, the registry, or a command line LÖVE would eat.
--  Values are percent-decoded, so ';' and '=' are safe.
-- ============================================================================

local Config = {}

-- timing / observation
Config.dt            = 1/30
Config.port          = 0
Config.port_file     = nil
Config.render        = "0"
Config.render_every  = 1
Config.grid_w        = 20
Config.grid_h        = 15
Config.max_bullets   = 24
Config.headless      = "1"
Config.log_level     = 1
Config.seed          = nil
Config.bosses        = nil
Config.boss_table    = {}
Config.max_steps     = 1800

-- reward weights
Config.r_alive        = 0.006
Config.r_damage       = 1.2       -- per point of player max-HP lost
Config.r_damage_dealt = 1.0       -- per unit of boss max-HP removed
Config.r_graze        = 0.01
Config.r_focus        = 0.0008    -- per frame spent focusing
Config.r_win          = 4.0
Config.r_death        = -3.0
Config.r_timeout      = -0.75

local NUMERIC = {
    dt = true, port = true, grid_w = true, grid_h = true, max_bullets = true,
    max_steps = true, log_level = true, seed = true, render_every = true,
    r_alive = true, r_damage = true, r_damage_dealt = true, r_graze = true,
    r_focus = true, r_win = true, r_death = true, r_timeout = true,
}

local BOOLEAN = {render = true, headless = true}

function Config.parse(envstr)
    for chunk in envstr:gmatch("[^;]+") do
        local k, v = chunk:match("^([^=]*)=(.*)$")
        if k then
            v = v:gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end)
            Config[k] = v
        end
    end

    for k in pairs(NUMERIC) do
        if Config[k] ~= nil then Config[k] = tonumber(Config[k]) end
    end
    -- Booleans arrive as "0"/"1"; normalise them because the string "0" is
    -- truthy in Lua and would silently enable every boolean flag.
    for k in pairs(BOOLEAN) do
        local v = Config[k]
        if type(v) == "string" then
            Config[k] = (v == "1" or v == "true" or v == "yes")
        elseif v == nil then
            Config[k] = false
        end
    end
    if Config.bosses and Config.bosses ~= "" then
        Config.boss_table = {}
        for name in Config.bosses:gmatch("[^,]+") do
            Config.boss_table[#Config.boss_table + 1] = name
        end
    end
    return Config
end

function Config.log(...)
    if (Config.log_level or 1) <= 0 then return end
    local parts = {}
    for i = 1, select("#", ...) do
        parts[#parts + 1] = tostring((select(i, ...)))
    end
    print("[rl] " .. table.concat(parts, " "))
    io.stdout:flush()
end

return Config