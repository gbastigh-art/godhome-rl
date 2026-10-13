-- ============================================================================
--  rl_net.lua -- blocking TCP endpoint for the harness.
--
--  LÖVE compiles LuaSocket into love.dll (>= 11.1), so `require("socket")`
--  works with no extra binaries.  We listen on 127.0.0.1; Python connects,
--  which keeps the handshake (and therefore error reporting) on the Python
--  side where a traceback is actually useful.
-- ============================================================================

local Net = {}

local socket = require("socket")

function Net.listen(port)
    local server = socket.bind("127.0.0.1", port or 0)
    local _, bound = server:getsockname()
    -- Block on accept(): the Python side is already waiting for the port file,
    -- and a short timeout here would race the trainer's connect().
    server:settimeout(nil)
    return server, bound
end

function Net.accept(server)
    local client = server:accept()
    if client then
        client:settimeout(30)   -- guard against a wedged trainer
        client:setoption("tcp-nodelay", true)
        client:setoption("keepalive", true)
    end
    return client
end

function Net.close(client)
    if client then
        pcall(function() client:close() end)
    end
end

return Net