-- A stand-in for the Garry's Mod client API, just faithful enough to run gmod_host.lua and the loader
-- (LuaJIT, like GMod). It enforces the rules that matter: render.* / cam.* / ToScreen only inside render hooks,
-- file access only below garrysmod/data. It does NOT prove the real API behaves this way: it catches typos,
-- nil indexing, wrong call order and wrong units in our code. Real-API mismatches are what `doctor` is for.
local Stub = { in_hook = false, view = nil, entities = {}, hooks = {}, calls = {}, opts = {} }

local V = {}
V.__index = V
function Vector(x, y, z) return setmetatable({ x = x or 0, y = y or 0, z = z or 0 }, V) end
V.__add = function(a, b) return Vector(a.x + b.x, a.y + b.y, a.z + b.z) end
V.__sub = function(a, b) return Vector(a.x - b.x, a.y - b.y, a.z - b.z) end
V.__mul = function(a, b)
  if type(a) == "number" then return Vector(a * b.x, a * b.y, a * b.z) end
  if type(b) == "number" then return Vector(a.x * b, a.y * b, a.z * b) end
  error("Vector * Vector is not supported by this stub")
end
V.__div = function(a, b) return Vector(a.x / b, a.y / b, a.z / b) end
function V:Distance(o) return math.sqrt((self.x - o.x) ^ 2 + (self.y - o.y) ^ 2 + (self.z - o.z) ^ 2) end
function V:Dot(o) return self.x * o.x + self.y * o.y + self.z * o.z end
function V:Length() return math.sqrt(self:Dot(self)) end
function V:Angle()
  local pitch = -math.deg(math.atan2(self.z, math.sqrt(self.x ^ 2 + self.y ^ 2)))
  local yaw = math.deg(math.atan2(self.y, self.x))
  return Angle(pitch, yaw, 0)
end
function V:ToScreen()
  local v = Stub.view
  if not v then error("ToScreen outside cam.Start3D") end
  local fwd, right, up = v.angles:Forward(), v.angles:Right(), v.angles:Up()
  local d = self - v.origin
  local z = d:Dot(fwd)
  if z <= 0.01 then return { x = 0, y = 0, visible = false } end
  -- Source "hor+ at 4:3": tan(vfov/2) = tan(fov/2) * 3/4
  local f = (v.h / 2) / (math.tan(math.rad(v.fov) / 2) * 0.75)
  return { x = v.w / 2 + f * d:Dot(right) / z, y = v.h / 2 - f * d:Dot(up) / z, visible = true }
end

local A = {}
A.__index = function(t, k)
  if k == "pitch" then return rawget(t, "p") elseif k == "yaw" then return rawget(t, "y")
  elseif k == "roll" then return rawget(t, "r") end
  return A[k]
end
function Angle(p, y, r) return setmetatable({ p = p or 0, y = y or 0, r = r or 0 }, A) end
local function basis(a)
  local sp, cp = math.sin(math.rad(a.p)), math.cos(math.rad(a.p))
  local sy, cy = math.sin(math.rad(a.y)), math.cos(math.rad(a.y))
  local sr, cr = math.sin(math.rad(a.r)), math.cos(math.rad(a.r))
  return Vector(cp * cy, cp * sy, -sp),
         Vector(-sr * sp * cy + cr * sy, -sr * sp * sy - cr * cy, -sr * cp),
         Vector(cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp)
end
function A:Forward() local f = basis(self) return f end
function A:Right() local _, r = basis(self) return r end
function A:Up() local _, _, u = basis(self) return u end

-- constants -----------------------------------------------------------------------------------
RENDERGROUP_OPAQUE, EF_BONEMERGE, MASK_SOLID_BRUSHONLY, MASK_SOLID, MASK_OPAQUE = 7, 1, 16395, 33570827, 16513
MATERIAL_FOG_LINEAR, RT_SIZE_LITERAL, MATERIAL_RT_DEPTH_SEPARATE, IMAGE_FORMAT_RGB888 = 1, 2, 3, 4
VERSION, SERVER, CLIENT = 240222, false, true
local holds = { "PISTOL", "SMG1", "AR2", "SHOTGUN", "MELEE", "CROSSBOW", "RPG" }
local stances = { "IDLE", "WALK", "RUN", "IDLE_CROUCH", "WALK_CROUCH" }
for s, sn in ipairs(stances) do
  _G["ACT_HL2MP_" .. sn] = s
  for h, hn in ipairs(holds) do _G["ACT_HL2MP_" .. sn .. "_" .. hn] = 10 * h + s end
end

-- misc globals ---------------------------------------------------------------------------------
function IsValid(e) return e ~= nil and type(e) == "table" and e._valid == true end
function SysTime() return Stub.clock() end
local frame_no = 0
function FrameNumber() return frame_no end
function AddCSLuaFile() end
function DrawColorModify(t) Stub.calls.color_modify = t end
function Stub.clock() return os.clock() end

-- hooks / concommands ----------------------------------------------------------------------------
hook = { Add = function(ev, id, fn) Stub.hooks[ev] = Stub.hooks[ev] or {}; Stub.hooks[ev][id] = fn end }
concommand = { Add = function(name, fn) Stub.calls["cmd_" .. name] = fn end }
function Stub.fire(ev, ...)
  local last
  for _, fn in pairs(Stub.hooks[ev] or {}) do last = fn(...) end
  return last
end

-- files (only garrysmod/data) --------------------------------------------------------------------
file = {}
local function dpath(p) return Stub.data_dir .. "/" .. p end
function file.Read(p, where)
  assert(where == "DATA", "this stub only knows the DATA path")
  local f = io.open(dpath(p), "rb")
  if not f then return nil end
  local s = f:read("*a")
  f:close()
  return s
end
function file.Write(p, data)
  assert(p:match("%.json$") or p:match("%.txt$") or p:match("%.dat$"), "extension not allowed in data/: " .. p)
  local f = assert(io.open(dpath(p), "wb"))
  f:write(data)
  f:close()
end
function file.CreateDir(p) os.execute("mkdir -p '" .. dpath(p) .. "'") end
function file.Open(p, mode, where)
  assert(where == "DATA")
  local f = io.open(dpath(p), mode)
  if not f then return nil end
  return { Write = function(_, d) f:write(d) end, Close = function() f:close() end }
end

util = {}
function util.JSONToTable(s)
  local out = {}
  for a, b, c in s:gmatch("%[%s*(-?[%d%.eE+-]+)%s*,%s*(-?[%d%.eE+-]+)%s*,%s*(-?[%d%.eE+-]+)%s*%]") do
    out[#out + 1] = { tonumber(a), tonumber(b), tonumber(c) }
  end
  return out
end
function util.TableToJSON(t)
  local parts = {}
  for _, p in ipairs(t) do parts[#parts + 1] = string.format("[%g,%g,%g]", p[1], p[2], p[3]) end
  return "[" .. table.concat(parts, ",") .. "]"
end
function util.TraceLine(t)
  Stub.calls.traces = (Stub.calls.traces or 0) + 1
  local s, e = t.start, t.endpos
  local res = { Hit = false, StartSolid = false, Fraction = 1, HitPos = e, HitNormal = Vector(0, 0, 1) }
  if Stub.opts.no_ground and t.mask == MASK_SOLID_BRUSHONLY and math.abs(e.z - s.z) > 100 then return res end
  if (s.z - 0) * (e.z - 0) < 0 then            -- crosses the ground plane z = 0
    local f = (s.z - 0) / (s.z - e.z)
    res = { Hit = true, StartSolid = false, Fraction = f, HitNormal = Vector(0, 0, 1),
            HitPos = Vector(s.x + (e.x - s.x) * f, s.y + (e.y - s.y) * f, 0) }
  end
  local w = Stub.opts.wall_x                     -- an opaque wall plane x = wall_x
  if w and t.mask == MASK_OPAQUE and (s.x - w) * (e.x - w) < 0 then
    local f = (s.x - w) / (s.x - e.x)
    if f < res.Fraction then res = { Hit = true, StartSolid = false, Fraction = f, HitNormal = Vector(-1, 0, 0),
                                     HitPos = Vector(w, s.y + (e.y - s.y) * f, s.z + (e.z - s.z) * f) } end
  end
  return res
end
function util.TraceHull(t) return { Hit = false, StartSolid = false, Fraction = 1 } end

function LocalPlayer() return { GetPos = function() return Vector(0, 0, 0) end } end

player_manager = {
  AllValidModels = function() return { kleiner = "models/player/kleiner.mdl", alyx = "models/player/alyx.mdl",
                                       police = "models/player/police.mdl" } end,
  TranslatePlayerModel = function(name) return "models/player/" .. name .. ".mdl" end,
}

-- render / cam (hook-only) ----------------------------------------------------------------------
local function need_hook(what)
  if not Stub.in_hook then error(what .. " called outside a render hook (GMod forbids this)", 2) end
end
render = {}
function render.PushRenderTarget(rt) need_hook("render.PushRenderTarget"); Stub.rt_depth = (Stub.rt_depth or 0) + 1 end
function render.PopRenderTarget() need_hook("render.PopRenderTarget"); Stub.rt_depth = Stub.rt_depth - 1 end
function render.Clear() need_hook("render.Clear") end
function render.RenderView(v)
  need_hook("render.RenderView")
  assert((Stub.rt_depth or 0) > 0, "RenderView outside a pushed render target")
  Stub.calls.render_view = v
  Stub.calls.render_views = (Stub.calls.render_views or 0) + 1
end
function render.Capture(t)
  need_hook("render.Capture")
  assert(t.format == "png" or t.format == "jpeg")
  return Stub.png
end
for _, n in ipairs({ "FogMode", "FogStart", "FogEnd", "FogMaxDensity", "FogColor" }) do
  render[n] = function(...) Stub.calls["fog_" .. n] = { ... } end
end
function GetRenderTargetEx(name, w, h) return { name = name, w = w, h = h } end
cam = {}
function cam.Start3D(origin, angles, fov, x, y, w, h)
  need_hook("cam.Start3D")
  Stub.view = { origin = origin, angles = angles, fov = fov, w = w, h = h }
end
function cam.End3D() need_hook("cam.End3D"); Stub.view = nil end

-- clientside models -----------------------------------------------------------------------------
local E = {}
E.__index = E
function ClientsideModel(path, group)
  local e = setmetatable({ _valid = true, path = path, pos = Vector(0, 0, 0), ang = Angle(0, 0, 0), seq = 0, cycle = 0,
                           nodraw = false, children = {}, pose = {}, skin = 0, groups = {} }, E)
  Stub.entities[#Stub.entities + 1] = e
  return e
end
function E:SetPos(p) self.pos = p end
function E:GetPos() return self.pos end
function E:SetAngles(a) self.ang = a end
function E:GetForward() return self.ang:Forward() end
function E:SetNoDraw(b) self.nodraw = b end
function E:Remove() self._valid = false end
function E:SetParent(p) self.parent = p; p.children[#p.children + 1] = self end
function E:AddEffects(f) self.effects = f end
function E:SkinCount() return 3 end
function E:SetSkin(n) self.skin = n end
function E:GetNumBodyGroups() return 2 end
function E:GetBodygroupCount(i) return 3 end
function E:SetBodygroup(i, v) self.groups[i] = v end
function E:SelectWeightedSequence(act) if act >= 1 and act < 100 then return act end return -1 end
function E:ResetSequence(s) self.seq = s end
function E:SetPlaybackRate(r) end
function E:SetCycle(c) self.cycle = c end
function E:SetPoseParameter(n, v) self.pose[n] = v end
function E:InvalidateBoneCache() end
function E:SetupBones() end
function E:LookupBone(name) for i, n in ipairs(self:_names()) do if n == name then return i - 1 end end return nil end
function E:GetRenderBounds() return Vector(-16, -16, 0), Vector(16, 16, 72) end
function E:LocalToWorld(v)
  local c, s = math.cos(math.rad(self.ang.y)), math.sin(math.rad(self.ang.y))
  return Vector(self.pos.x + c * v.x - s * v.y, self.pos.y + s * v.x + c * v.y, self.pos.z + v.z)
end

local BONES = {  -- name, local (x fwd, y left, z up) in METERS, per ValveBiped
  { "ValveBiped.Bip01_Pelvis", 0, 0, 0.95 }, { "ValveBiped.Bip01_Spine", 0, 0, 1.1 },
  { "ValveBiped.Bip01_Neck1", 0, 0, 1.5 }, { "ValveBiped.Bip01_Head1", 0, 0, 1.7 },
  { "ValveBiped.Bip01_L_Clavicle", 0, 0.1, 1.45 }, { "ValveBiped.Bip01_R_Clavicle", 0, -0.1, 1.45 },
  { "ValveBiped.Bip01_L_UpperArm", 0, 0.2, 1.45 }, { "ValveBiped.Bip01_R_UpperArm", 0, -0.2, 1.45 },
  { "ValveBiped.Bip01_L_Forearm", "sw", 0.26, 1.15 }, { "ValveBiped.Bip01_R_Forearm", "-sw", -0.26, 1.15 },
  { "ValveBiped.Bip01_L_Hand", "2sw", 0.28, 0.9 }, { "ValveBiped.Bip01_R_Hand", "-2sw", -0.28, 0.9 },
  { "ValveBiped.Bip01_L_Thigh", 0, 0.09, 0.92 }, { "ValveBiped.Bip01_R_Thigh", 0, -0.09, 0.92 },
  { "ValveBiped.Bip01_L_Calf", "-sw", 0.1, 0.5 }, { "ValveBiped.Bip01_R_Calf", "sw", -0.1, 0.5 },
  { "ValveBiped.Bip01_L_Foot", "-2sw", 0.1, 0.08 }, { "ValveBiped.Bip01_R_Foot", "2sw", -0.1, 0.08 },
}
function E:_names()
  local n = {}
  for i, b in ipairs(BONES) do n[i] = b[1] end
  return n
end
function E:GetBoneCount() return #BONES end
function E:GetBoneName(i) return BONES[i + 1][1] end
function E:GetBonePosition(i)
  local b = BONES[i + 1]
  local stance = self.seq % 10
  local sw = (stance == 2 or stance == 5) and 0.25 * math.sin(2 * math.pi * self.cycle) or 0
  local crouch = (stance == 4 or stance == 5) and 0.35 or 0
  local function val(v) if v == "sw" then return sw elseif v == "-sw" then return -sw elseif v == "2sw" then return 2 * sw
    elseif v == "-2sw" then return -2 * sw end return v end
  local x, y, z = val(b[2]), b[3], b[4]
  if z > 0.3 and i ~= 0 and b[1] ~= "ValveBiped.Bip01_L_Foot" and b[1] ~= "ValveBiped.Bip01_R_Foot" then z = z - crouch end
  local c, s = math.cos(math.rad(self.ang.y)), math.sin(math.rad(self.ang.y))
  local inv = 1 / 0.01905
  return Vector((self.pos.x * 0.01905 + c * x - s * y) * inv, (self.pos.y * 0.01905 + s * x + c * y) * inv,
                (self.pos.z * 0.01905 + z) * inv), Angle(0, 0, 0)
end
function E:GetBoneMatrix(i) return nil end

-- driver ----------------------------------------------------------------------------------------
function Stub.setup(data_dir, png, opts)
  Stub.data_dir, Stub.png, Stub.opts = data_dir, png, opts or {}
  os.execute("mkdir -p '" .. data_dir .. "'")
end
function Stub.tick()
  frame_no = frame_no + 1
  Stub.fire("Think")
  Stub.in_hook = true
  local ok, err = pcall(Stub.fire, "PostRender")
  Stub.in_hook = false
  Stub.rt_depth, Stub.view = 0, nil
  if not ok then error(err, 0) end
end
function Stub.include(path, root)
  local map = { ["dataopen/dataopen_rpc.lua"] = Stub.runtime_path }
  local full = map[path] or (root .. "/" .. path)
  return dofile(full)
end
return Stub
