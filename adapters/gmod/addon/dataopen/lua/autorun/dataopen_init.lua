-- DataOpen for Garry's Mod: loader. Runs in a map as singleplayer or listen server; start the game with
-- -insecure on a local server. Data folder: garrysmod/data/dataopen/ (mailbox + staging).
if SERVER then
  AddCSLuaFile("dataopen/dataopen_rpc.lua")
  AddCSLuaFile("dataopen/gmod_host.lua")
  return
end

-- Create the mailbox folder right away so it is visible (and the core can point at it) before the first request.
file.CreateDir("dataopen")
file.CreateDir("dataopen/staging")

local R = include("dataopen/dataopen_rpc.lua")
local host = include("dataopen/gmod_host.lua")(R)
local rt = R.new(host, { log = function(m) print("[dataopen] " .. m) end })

-- One poll per rendered frame. Do not open the pause menu while collecting: Think stops there.
hook.Add("Think", "dataopen_poll", function()
  local ok, err = pcall(rt.poll, rt)
  if not ok then print("[dataopen] poll error: " .. tostring(err)) end
end)

concommand.Add("dataopen_status", function()
  print("[dataopen] protocol " .. R.PROTOCOL .. ", last request id " .. tostring(rt.last_id)
        .. ", frames " .. tostring(rt.state.frames) .. ", mailbox garrysmod/data/dataopen/")
end)

-- Stand somewhere interesting and run `dataopen_mark`: the spot is added to data/dataopen/locations.json and
-- scenes cycle through the marked spots (more varied backgrounds than roaming around one place).
concommand.Add("dataopen_mark", function()
  file.CreateDir("dataopen")
  local list = util.JSONToTable(file.Read("dataopen/locations.json", "DATA") or "[]") or {}
  local p = LocalPlayer():GetPos()
  list[#list + 1] = { p.x, p.y, p.z }
  file.Write("dataopen/locations.json", util.TableToJSON(list))
  print("[dataopen] marked location #" .. #list)
end)

print("[dataopen] ready. Mailbox: garrysmod/data/dataopen/  Commands: dataopen_status, dataopen_mark")
