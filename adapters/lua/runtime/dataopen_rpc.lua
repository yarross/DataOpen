-- DataOpen Lua runtime: speaks protocol v1 (docs/PROTOCOL.md) from inside a game, over the file mailbox.
--
-- Game-independent. A game mod supplies a `host` table (engine bindings, see docs/MODDING.md) and calls
-- `rt:poll()` once per game tick. Works on Lua 5.1 / LuaJIT (Garry's Mod, Cyber Engine Tweaks) and 5.4 (UE4SS).
-- Restrictions kept on purpose: no `//`, no bit ops, no `goto`, no integer-only APIs.
--
-- Handlers run inside coroutines, so engine code can be written linearly: `rt:wait_frames(n)` yields until
-- n more ticks have passed, `rt:wait_until(pred, timeout)` waits for an asynchronous engine result.

local R = {}
R.PROTOCOL = 1
local unpack = table.unpack or unpack

---------------------------------------------------------------------------------------------------
-- JSON
---------------------------------------------------------------------------------------------------
local J = {}
R.json = J

function J.array(t) return setmetatable(t or {}, { __jsontype = "array" }) end
function J.object(t) return setmetatable(t or {}, { __jsontype = "object" }) end

local escapes = { ['"'] = '\\"', ['\\'] = '\\\\', ['\b'] = '\\b', ['\f'] = '\\f',
                  ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t' }

local function encode_string(s)
  local out = s:gsub('[%c"\\]', function(c)
    return escapes[c] or string.format("\\u%04x", c:byte())
  end)
  return '"' .. out .. '"'
end

local function encode_number(n)
  if n ~= n or n == math.huge or n == -math.huge then error("cannot encode a non-finite number") end
  if n == math.floor(n) and math.abs(n) < 1e15 then return string.format("%d", n) end
  return string.format("%.9g", n)
end

local function is_array(t)
  local mt = getmetatable(t)
  if mt and mt.__jsontype == "array" then return true end
  if mt and mt.__jsontype == "object" then return false end
  if next(t) == nil then return false end
  local n = 0
  for k in pairs(t) do
    if type(k) ~= "number" then return false end
    n = n + 1
  end
  for i = 1, n do if t[i] == nil then return false end end
  return true
end

local function encode(v, out)
  local tv = type(v)
  if tv == "nil" then out[#out + 1] = "null"
  elseif tv == "boolean" then out[#out + 1] = v and "true" or "false"
  elseif tv == "number" then out[#out + 1] = encode_number(v)
  elseif tv == "string" then out[#out + 1] = encode_string(v)
  elseif tv == "table" then
    if is_array(v) then
      out[#out + 1] = "["
      for i = 1, #v do
        if i > 1 then out[#out + 1] = "," end
        encode(v[i], out)
      end
      out[#out + 1] = "]"
    else
      local keys = {}
      for k, val in pairs(v) do
        if val ~= nil then keys[#keys + 1] = tostring(k) end
      end
      table.sort(keys)
      out[#out + 1] = "{"
      for i, k in ipairs(keys) do
        if i > 1 then out[#out + 1] = "," end
        out[#out + 1] = encode_string(k) .. ":"
        local val = v[k]
        if val == nil then val = v[tonumber(k)] end
        encode(val, out)
      end
      out[#out + 1] = "}"
    end
  else
    error("cannot encode a " .. tv)
  end
end

function J.encode(v)
  local out = {}
  encode(v, out)
  return table.concat(out)
end

local function utf8_char(cp)
  if cp < 0x80 then return string.char(cp)
  elseif cp < 0x800 then return string.char(0xC0 + math.floor(cp / 64), 0x80 + cp % 64)
  elseif cp < 0x10000 then
    return string.char(0xE0 + math.floor(cp / 4096), 0x80 + math.floor(cp / 64) % 64, 0x80 + cp % 64)
  end
  return string.char(0xF0 + math.floor(cp / 262144), 0x80 + math.floor(cp / 4096) % 64,
                     0x80 + math.floor(cp / 64) % 64, 0x80 + cp % 64)
end

local function skip_ws(s, i)
  return s:find("[^ \t\r\n]", i) or (#s + 1)
end

local decode_value

local function decode_string(s, i)
  local out, j = {}, i + 1
  while true do
    local c = s:sub(j, j)
    if c == "" then error("unterminated string at " .. i) end
    if c == '"' then return table.concat(out), j + 1 end
    if c == "\\" then
      local e = s:sub(j + 1, j + 1)
      local map = { b = "\b", f = "\f", n = "\n", r = "\r", t = "\t", ['"'] = '"', ["\\"] = "\\", ["/"] = "/" }
      if e == "u" then
        local hex = s:sub(j + 2, j + 5)
        if not hex:match("^%x%x%x%x$") then error("bad \\u escape at " .. j) end
        local cp = tonumber(hex, 16)
        j = j + 6
        if cp >= 0xD800 and cp <= 0xDBFF and s:sub(j, j + 1) == "\\u" then
          local lo = tonumber(s:sub(j + 2, j + 5), 16)
          if lo and lo >= 0xDC00 and lo <= 0xDFFF then
            cp = 0x10000 + (cp - 0xD800) * 1024 + (lo - 0xDC00)
            j = j + 6
          end
        end
        out[#out + 1] = utf8_char(cp)
      elseif map[e] then
        out[#out + 1] = map[e]
        j = j + 2
      else
        error("bad escape at " .. j)
      end
    else
      local k = s:find('["\\]', j) or (#s + 1)
      out[#out + 1] = s:sub(j, k - 1)
      j = k
    end
  end
end

decode_value = function(s, i)
  i = skip_ws(s, i)
  local c = s:sub(i, i)
  if c == "{" then
    local obj = {}
    i = skip_ws(s, i + 1)
    if s:sub(i, i) == "}" then return obj, i + 1 end
    while true do
      i = skip_ws(s, i)
      if s:sub(i, i) ~= '"' then error("expected a string key at " .. i) end
      local k
      k, i = decode_string(s, i)
      i = skip_ws(s, i)
      if s:sub(i, i) ~= ":" then error("expected ':' at " .. i) end
      local v
      v, i = decode_value(s, i + 1)
      obj[k] = v
      i = skip_ws(s, i)
      local d = s:sub(i, i)
      if d == "}" then return obj, i + 1 end
      if d ~= "," then error("expected ',' or '}' at " .. i) end
      i = i + 1
    end
  elseif c == "[" then
    local arr, n = J.array({}), 0
    i = skip_ws(s, i + 1)
    if s:sub(i, i) == "]" then return arr, i + 1 end
    while true do
      local v
      v, i = decode_value(s, i)
      n = n + 1
      arr[n] = v
      i = skip_ws(s, i)
      local d = s:sub(i, i)
      if d == "]" then return arr, i + 1 end
      if d ~= "," then error("expected ',' or ']' at " .. i) end
      i = i + 1
    end
  elseif c == '"' then
    return decode_string(s, i)
  elseif s:sub(i, i + 3) == "true" then return true, i + 4
  elseif s:sub(i, i + 4) == "false" then return false, i + 5
  elseif s:sub(i, i + 3) == "null" then return nil, i + 4
  else
    local num = s:match("^-?%d+%.?%d*[eE]?[+-]?%d*", i)
    if not num or num == "" then error("unexpected character at " .. i) end
    return tonumber(num), i + #num
  end
end

function J.decode(s)
  local v, i = decode_value(s, 1)
  i = skip_ws(s, i)
  if i <= #s then error("trailing data at " .. i) end
  return v
end

---------------------------------------------------------------------------------------------------
-- Bone maps shipped for common rigs (keypoint -> { {bone, weight}, ... }); mods may override per game
---------------------------------------------------------------------------------------------------
R.maps = {}

-- Valve "ValveBiped" rig (Source engine: Garry's Mod playermodels, HL2 NPCs)
do
  local function lr(kp_l, kp_r, name)
    return { [kp_l] = { { "ValveBiped.Bip01_L_" .. name, 1.0 } }, [kp_r] = { { "ValveBiped.Bip01_R_" .. name, 1.0 } } }
  end
  local m = {
    head = { { "ValveBiped.Bip01_Head1", 1.0 } },
    neck = { { "ValveBiped.Bip01_Neck1", 1.0 } },
    pelvis = { { "ValveBiped.Bip01_Pelvis", 1.0 } },
  }
  for _, part in ipairs({ { "l_shoulder", "r_shoulder", "UpperArm" }, { "l_elbow", "r_elbow", "Forearm" },
                          { "l_wrist", "r_wrist", "Hand" }, { "l_knee", "r_knee", "Calf" },
                          { "l_ankle", "r_ankle", "Foot" } }) do
    for k, v in pairs(lr(part[1], part[2], part[3])) do m[k] = v end
  end
  R.maps.valvebiped = m
end

-- Unreal Engine mannequin / Metahuman-style names
do
  local m = {
    head = { { "head", 1.0 } }, neck = { { "neck_01", 1.0 } }, pelvis = { { "pelvis", 1.0 } },
  }
  local pairs_ = { { "shoulder", "upperarm" }, { "elbow", "lowerarm" }, { "wrist", "hand" },
                   { "knee", "calf" }, { "ankle", "foot" } }
  for _, p in ipairs(pairs_) do
    m["l_" .. p[1]] = { { p[2] .. "_l", 1.0 } }
    m["r_" .. p[1]] = { { p[2] .. "_r", 1.0 } }
  end
  R.maps.ue_mannequin = m
end

---------------------------------------------------------------------------------------------------
-- Runtime
---------------------------------------------------------------------------------------------------
local Runtime = {}
Runtime.__index = Runtime

local function norm(s) return (tostring(s):lower():gsub("[^%w]", "")) end

local function xyz(p)
  if p == nil then return nil end
  if p.x ~= nil then return { x = p.x, y = p.y, z = p.z } end
  return { x = p[1], y = p[2], z = p[3] }
end

local function add(a, b, k)
  k = k or 1
  return { x = a.x + b.x * k, y = a.y + b.y * k, z = a.z + b.z * k }
end

function R.new(host, opts)
  opts = opts or {}
  local self = setmetatable({}, Runtime)
  self.host = host
  self.timeout_s = opts.timeout_s or 120          -- hard cap for one request
  self.freeze_timeout_s = opts.freeze_timeout_s or 60  -- unfreeze if the core vanishes mid-frame
  self.options = {}
  self.bone_map = {}                              -- per-keypoint overrides from the profile
  self.keypoints = {}
  self.state = { handles = {}, frames = 0, frozen = false }
  self.last_id = nil
  self.co = nil
  self.handlers = R.handlers
  self.log = opts.log or function(msg) end
  return self
end

function Runtime:scale() return self.host.unit_scale or 1 end

function Runtime:reply(req, result, err)
  local msg
  if err then
    msg = { id = req.id, error = { message = tostring(err.message or err), type = err.type or "ModError" } }
  else
    msg = { id = req.id, result = result or J.object({}) }
  end
  local ok, text = pcall(J.encode, msg)
  if not ok then
    text = J.encode({ id = req.id, error = { message = "cannot encode the response: " .. tostring(text), type = "EncodeError" } })
  end
  self.host.fs.write("res.json", text)
end

function Runtime:wait_frames(n)
  if n <= 0 then return end
  self.wait_left = n - 1
  coroutine.yield()
end

function Runtime:wait_until(pred, timeout_s)
  if pred() then return true end
  self.wait_pred = pred
  self.wait_deadline = self.host.now() + (timeout_s or 10)
  self.wait_timed_out = false
  coroutine.yield()
  return not self.wait_timed_out
end

function Runtime:_finish(result, err)
  self:reply(self.req, result, err)
  self.co, self.req, self.wait_left, self.wait_pred = nil, nil, nil, nil
end

function Runtime:_resume(now)
  if self.wait_left and self.wait_left > 0 then
    self.wait_left = self.wait_left - 1
    return
  end
  if self.wait_pred then
    if self.wait_pred() then
      self.wait_pred = nil
    elseif now > self.wait_deadline then
      self.wait_pred, self.wait_timed_out = nil, true
    else
      return
    end
  end
  if now - self.started > self.timeout_s then
    self:_finish(nil, { message = "handler timed out after " .. self.timeout_s .. "s", type = "Timeout" })
    return
  end
  local ok, res = coroutine.resume(self.co)
  if not ok then
    self:_finish(nil, { message = tostring(res) .. "\n" .. tostring(debug.traceback(self.co)), type = "LuaError" })
  elseif coroutine.status(self.co) == "dead" then
    self:_finish(res)
  end
end

function Runtime:unfreeze()
  if self.state.frozen then
    self.state.frozen = false
    if self.host.unfreeze then self.host.unfreeze() end
  end
end

--- Call once per game tick.
function Runtime:poll()
  local now = self.host.now()
  if self.co then
    self:_resume(now)
    return
  end
  if self.state.frozen and now - (self.state.frozen_at or now) > self.freeze_timeout_s then
    self.log("core vanished while the game was frozen: unfreezing")
    self:unfreeze()
  end
  local raw = self.host.fs.read("req.json")
  if not raw or raw == "" then return end
  local ok, req = pcall(J.decode, raw)
  if not ok or type(req) ~= "table" or req.id == nil then return end  -- half-written: try again next tick
  if req.id == self.last_id then return end
  self.last_id = req.id
  if req.v ~= R.PROTOCOL then
    self:reply(req, nil, { message = "protocol version " .. tostring(req.v) .. " != " .. R.PROTOCOL, type = "ProtocolError" })
    return
  end
  local h = self.handlers[req.method]
  if not h then
    self:reply(req, nil, { message = "unknown method " .. tostring(req.method), type = "ProtocolError" })
    return
  end
  self.req, self.started = req, now
  self.co = coroutine.create(function() return h(self, req.params or {}) end)
  self:_resume(now)
end

---------------------------------------------------------------------------------------------------
-- Bone resolution
---------------------------------------------------------------------------------------------------
function Runtime:rig_map(rig_id)
  local base = self.host.bone_map and self.host.bone_map(rig_id) or {}
  local merged = {}
  for k, v in pairs(base) do merged[k] = v end
  for k, v in pairs(self.bone_map) do merged[k] = v end
  return merged
end

--- bones: { name = {x,y,z} } in ENGINE units. Returns flat meters list, validity list, unmapped keypoint names.
function Runtime:resolve(rig_id, bones)
  local lookup = {}
  for name, pos in pairs(bones) do lookup[norm(name)] = xyz(pos) end
  local map, s = self:rig_map(rig_id), self:scale()
  local flat, valid, unmapped = J.array({}), J.array({}), {}
  for _, kp in ipairs(self.keypoints) do
    local parts, sum, wsum, ok = map[kp], { x = 0, y = 0, z = 0 }, 0, true
    if not parts or #parts == 0 then ok = false end
    if ok then
      for _, part in ipairs(parts) do
        local p = lookup[norm(part[1])]
        if not p then ok = false break end
        local w = part[2] or 1
        sum = add(sum, p, w)
        wsum = wsum + w
      end
    end
    if ok then
      flat[#flat + 1] = sum.x / wsum * s
      flat[#flat + 1] = sum.y / wsum * s
      flat[#flat + 1] = sum.z / wsum * s
      valid[#valid + 1] = true
    else
      flat[#flat + 1] = 0
      flat[#flat + 1] = 0
      flat[#flat + 1] = 0
      valid[#valid + 1] = false
      unmapped[#unmapped + 1] = kp
    end
  end
  return flat, valid, unmapped
end

---------------------------------------------------------------------------------------------------
-- Handlers
---------------------------------------------------------------------------------------------------
local H = {}
R.handlers = H

local function bone_names(bones)
  local names = {}
  for name in pairs(bones) do names[#names + 1] = name end
  table.sort(names)
  return names
end

function H.hello(rt, p)
  local host = rt.host
  rt.options = p.options or {}
  rt.bone_map = p.bone_map or {}
  rt.keypoints = {}
  local kps = (p.schema and p.schema.keypoints) or {}
  for i = 1, #kps do rt.keypoints[i] = kps[i] end
  rt.state = { handles = {}, frames = 0, frozen = false }
  rt:unfreeze()
  if host.init then host.init(rt.options, rt, p.image) end

  local errors = J.array({})
  for _, rig in ipairs(host.rigs and host.rigs() or {}) do
    local map = rt:rig_map(rig)
    for _, kp in ipairs(rt.keypoints) do
      if not map[kp] then errors[#errors + 1] = "rig " .. rig .. " has no bone mapping for keypoint " .. kp end
    end
  end
  local caps = J.array({ "probes" })
  if host.visibility then caps[#caps + 1] = "engine_visibility" end
  if host.entity_extras then caps[#caps + 1] = "hull_points" end
  if host.capture_image then caps[#caps + 1] = "image_engine" end
  local w, h = 1280, 720
  if host.render_size then w, h = host.render_size() end
  return {
    protocol = R.PROTOCOL, game = host.name, engine = host.engine,
    game_version = host.game_version and host.game_version() or "unknown",
    mod_version = host.mod_version or "0", capabilities = caps,
    image = { width = w, height = h }, schema_errors = errors,
    parameter_space = host.parameter_space and host.parameter_space() or {},
  }
end

function H.begin_scene(rt, p)
  local host = rt.host
  local handles = host.begin_scene(p.scene, rt) or {}
  rt.state.handles = handles
  local out = J.array({})
  for _, h in ipairs(handles) do
    out[#out + 1] = { entity_id = h.entity_id, rig_id = h.rig_id, meta = h.meta or J.object({}) }
  end
  return { handles = out }
end

local function head_index(rt)
  for i, kp in ipairs(rt.keypoints) do if kp == "head" then return i end end
  return 1
end

function Runtime:make_probes(cam_m, skeletons, w, h)
  local host, s = self.host, self:scale()
  local pts = {
    add(add(cam_m.pos, cam_m.forward, 6), cam_m.right, 1.5),
    add(add(add(cam_m.pos, cam_m.forward, 12), cam_m.right, -3), cam_m.up, 2),
    add(add(cam_m.pos, cam_m.forward, 4), cam_m.up, 1.2),
  }
  local hi = head_index(self)
  for i = 1, math.min(3, #skeletons) do
    local sk = skeletons[i]
    local o = (hi - 1) * 3
    pts[#pts + 1] = { x = sk[o + 1], y = sk[o + 2], z = sk[o + 3] }
  end
  local eng = {}
  for i, pt in ipairs(pts) do eng[i] = { x = pt.x / s, y = pt.y / s, z = pt.z / s } end
  local uvs = {}
  if host.project_batch then
    -- engines that can only project inside a render hook (Garry's Mod ToScreen) answer in one batch
    uvs = host.project_batch(eng, w, h, self) or {}
  else
    for i = 1, #eng do uvs[i] = host.project(eng[i], w, h) or false end
  end
  local probes = J.array({})
  for i, pt in ipairs(pts) do
    local uv = uvs[i]
    probes[#probes + 1] = { world = J.array({ pt.x, pt.y, pt.z }),
                            screen = uv and J.array({ uv.x or uv[1], uv.y or uv[2] }) or nil }
  end
  return probes
end

function H.capture_frame(rt, p)
  local host, st, s = rt.host, rt.state, rt:scale()
  rt:unfreeze()
  host.set_active(st.handles, p.active ~= false, rt)
  if host.apply_frame then host.apply_frame(p.frame, st.handles, rt) end
  host.place_camera(p.frame.camera, st.handles, p.width, p.height, rt)
  rt:wait_frames(rt.options.settle_ticks or 2)
  host.freeze()
  st.frozen, st.frozen_at = true, host.now()
  rt:wait_frames(1)

  local c = host.read_camera(p.width, p.height)
  local cam = { pos = { x = c.pos.x * s, y = c.pos.y * s, z = c.pos.z * s }, forward = xyz(c.forward),
                right = xyz(c.right), up = xyz(c.up) }
  local entities, skeletons, warnings = J.array({}), {}, J.array({})
  for _, h in ipairs(host.entities(rt)) do
    local bones = host.read_bones(h)
    local flat, valid, unmapped = rt:resolve(h.rig_id, bones)
    local e = { entity_id = h.entity_id, rig_id = h.rig_id, skeleton_world = flat, joint_valid = valid,
                meta = h.meta and J.object(h.meta) or J.object({}) }
    if #unmapped > 0 then
      warnings[#warnings + 1] = "entity " .. h.entity_id .. ": unmapped keypoints " .. table.concat(unmapped, ",")
    end
    if host.visibility then
      local vis = J.array({})
      for i = 1, #rt.keypoints do
        local pt = { x = flat[(i - 1) * 3 + 1] / s, y = flat[(i - 1) * 3 + 2] / s, z = flat[(i - 1) * 3 + 3] / s }
        vis[i] = (valid[i] and host.visibility(c.pos, pt, h)) and 2 or 1
      end
      e.engine_visibility = vis
    end
    if host.entity_extras then
      local x = host.entity_extras(h) or {}
      if x.forward then e.meta.forward = J.array({ x.forward.x or x.forward[1], x.forward.y or x.forward[2],
                                                    x.forward.z or x.forward[3] }) end
      if x.hull_points then
        local hp = J.array({})
        for _, q in ipairs(x.hull_points) do
          hp[#hp + 1] = q.x * s
          hp[#hp + 1] = q.y * s
          hp[#hp + 1] = q.z * s
        end
        e.hull_points = hp
      end
    end
    entities[#entities + 1] = e
    skeletons[#skeletons + 1] = flat
  end

  if p.image_mode == "engine" and host.capture_image then
    host.capture_image(p.frame_id, p.width, p.height, rt)
  end
  st.frames = st.frames + 1
  local fov = c.fov_v_deg and { fov_v_deg = c.fov_v_deg } or { fov_h_deg = c.fov_h_deg }
  return {
    frame_token = p.frame_id, tick = host.tick and host.tick() or st.frames,
    camera = { width = p.width, height = p.height, pos = J.array({ cam.pos.x, cam.pos.y, cam.pos.z }),
               forward = J.array({ cam.forward.x, cam.forward.y, cam.forward.z }),
               right = J.array({ cam.right.x, cam.right.y, cam.right.z }),
               up = J.array({ cam.up.x, cam.up.y, cam.up.z }),
               fov_v_deg = fov.fov_v_deg, fov_h_deg = fov.fov_h_deg, near = c.near or 0.1 },
    entities = entities, probes = rt:make_probes(cam, skeletons, p.width, p.height), warnings = warnings,
  }
end

function H.release(rt, p)
  rt:unfreeze()
  return J.object({})
end

function H.commit(rt, p)
  local staged = rt.host.commit_image(p.frame_token, p.dest, rt)
  rt:unfreeze()
  if staged then return { staged = staged } end
  return J.object({})
end

function H.discard(rt, p)
  if rt.host.discard_image then rt.host.discard_image(p.frame_token) end
  rt:unfreeze()
  return J.object({})
end

function H.end_scene(rt, p)
  rt:unfreeze()
  if rt.host.end_scene then rt.host.end_scene(rt) end
  rt.state.handles = {}
  return J.object({})
end

function H.health(rt, p)
  return { ok = true, frames = rt.state.frames, frozen = rt.state.frozen }
end

function H.shutdown(rt, p)
  rt:unfreeze()
  if rt.host.shutdown then rt.host.shutdown() end
  return J.object({})
end

function H.selftest(rt, p)
  local host = rt.host
  local checks = J.array({})
  local function add_check(name, ok, detail, hint, data)
    checks[#checks + 1] = { name = name, ok = ok, detail = detail, hint = hint, data = data }
  end
  if host.sample_bones then
    local sample = host.sample_bones(rt)
    if sample and sample.bones then
      local _, valid, unmapped = rt:resolve(sample.rig_id, sample.bones)
      add_check("bone_mapping", #unmapped == 0,
                (#valid - #unmapped) .. "/" .. #valid .. " keypoints resolved for rig " .. tostring(sample.rig_id),
                #unmapped > 0 and "add overrides to the profile's [bones] using the names in `bones_found`" or nil,
                { unmapped = J.array(unmapped), bones_found = J.array(bone_names(sample.bones)) })
    else
      add_check("bone_mapping", false, "no sample character available",
                "start a map with at least one humanoid, or set population_mode = 'observe' near NPCs")
    end
  end
  if host.selftests then
    for _, c in ipairs(host.selftests(rt) or {}) do
      add_check(c.name, c.ok, c.detail or "", c.hint, c.data)
    end
  end
  return { checks = checks }
end

return R
