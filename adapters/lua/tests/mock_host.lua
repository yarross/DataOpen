-- A simulated game ("host") for testing dataopen_rpc.lua without any real game.
-- Source-style conventions on purpose: Z-up world, engine units = inches (unit_scale), ValveBiped bone names.
-- Usage: build = dofile("mock_host.lua"); host = build(R, mailbox_dir, opts)
return function(R, dir, opts)
  opts = opts or {}
  local us = opts.unit_scale or 0.0254          -- meters per engine unit
  local inv = 1 / us
  local W, H = opts.width or 160, opts.height or 90
  local host = { name = "luamock", engine = "lua-sim", mod_version = "lua-test", unit_scale = us }
  local world = { actors = {}, active = true, tick = 0, frozen = false, cam = nil, pending = {}, unfreezes = 0 }
  host.world = world
  host.now = opts.now or os.clock

  host.fs = {
    read = function(name)
      local f = io.open(dir .. "/" .. name, "rb")
      if not f then return nil end
      local s = f:read("*a")
      f:close()
      return s
    end,
    write = function(name, data)
      local f = assert(io.open(dir .. "/" .. name, "wb"))
      f:write(data)
      f:close()
      return true
    end,
  }

  local function lcg(seed)
    local s = seed % 2147483647
    if s <= 0 then s = s + 2147483646 end
    return function()
      s = (s * 48271) % 2147483647
      return s / 2147483647
    end
  end
  local function norm3(v)
    local n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return { x = v.x / n, y = v.y / n, z = v.z / n }
  end
  local function cross(a, b)
    return { x = a.y * b.z - a.z * b.y, y = a.z * b.x - a.x * b.z, z = a.x * b.y - a.y * b.x }
  end
  local function dot(a, b) return a.x * b.x + a.y * b.y + a.z * b.z end

  local function bone_names()
    local p = "ValveBiped.Bip01_"
    if opts.alt_names then p = "Alt_" end
    return {
      head = p .. "Head1", neck = p .. "Neck1", pelvis = p .. "Pelvis",
      ls = p .. "L_UpperArm", rs = p .. "R_UpperArm", le = p .. "L_Forearm", re = p .. "R_Forearm",
      lw = p .. "L_Hand", rw = p .. "R_Hand", lk = p .. "L_Calf", rk = p .. "R_Calf",
      la = p .. "L_Foot", ra = p .. "R_Foot",
    }
  end

  local function bones_for(a)
    local f = a.frame or {}
    local walk = f.animation == "walk"
    local sw = walk and 0.25 * math.sin(2 * math.pi * (f.phase or 0)) or 0
    local L = {
      head = { 0, 0, 1.70 }, neck = { 0, 0, 1.50 }, pelvis = { 0, 0, 0.95 },
      ls = { 0, 0.20, 1.45 }, rs = { 0, -0.20, 1.45 }, le = { sw, 0.26, 1.15 }, re = { -sw, -0.26, 1.15 },
      lw = { 2 * sw, 0.28, 0.90 }, rw = { -2 * sw, -0.28, 0.90 },
      lk = { -sw, 0.10, 0.50 }, rk = { sw, -0.10, 0.50 }, la = { -2 * sw, 0.10, 0.08 }, ra = { 2 * sw, -0.10, 0.08 },
    }
    local c, s = math.cos(a.yaw), math.sin(a.yaw)
    local out, names = {}, bone_names()
    for key, name in pairs(names) do
      local p = L[key]
      out[name] = { x = (c * p[1] - s * p[2] + a.x) * inv, y = (s * p[1] + c * p[2] + a.y) * inv, z = p[3] * inv }
    end
    return out
  end

  function host.game_version() return "sim-1.0" end
  function host.render_size() return W, H end
  function host.rigs() return { "valve" } end
  function host.bone_map(rig) return R.maps.valvebiped end

  function host.parameter_space()
    return {
      environment = {},
      actor = {
        rig = { type = "categorical", choices = R.json.array({ "valve" }) },
        outfit = { type = "categorical", choices = R.json.array({ "casual", "armor" }) },
      },
      actor_frame = {
        animation = { type = "categorical", choices = R.json.array({ "stand", "walk" }) },
        phase = { type = "uniform", lo = 0, hi = 1 },
      },
    }
  end

  function host.init(options, rt)
    world.options = options
    world.actors, world.active = {}, true
  end

  function host.begin_scene(scene, rt)
    if host.fail_next == "begin_scene" then
      host.fail_next = nil
      error("simulated spawn failure")
    end
    local rnd = lcg(scene.seed)
    world.actors = {}
    local handles = {}
    for i, params in ipairs(scene.actors) do
      local r, th = scene.area.radius * math.sqrt(rnd()), rnd() * 2 * math.pi
      world.actors[i] = { x = r * math.cos(th), y = r * math.sin(th), yaw = rnd() * 2 * math.pi }
      handles[i] = { entity_id = i - 1, rig_id = "valve", meta = { outfit = params.outfit } }
    end
    world.active = true
    return handles
  end

  function host.set_active(handles, active) world.active = active end

  function host.apply_frame(frame, handles)
    for i, a in ipairs(world.actors) do a.frame = frame.actor_frame[i] end
  end

  function host.place_camera(spec, handles, w, h, rt)
    local tgt = world.actors[spec.target_index + 1] or world.actors[1]
    local target = { x = tgt.x, y = tgt.y, z = 1.0 }
    local y, p = math.rad(spec.yaw_deg), math.rad(spec.pitch_deg)
    local pos = { x = target.x + spec.distance * math.cos(p) * math.cos(y),
                  y = target.y + spec.distance * math.cos(p) * math.sin(y),
                  z = target.z + spec.distance * math.sin(p) }
    pos.z = math.max(pos.z + spec.height_offset, 0.3)
    local f = norm3({ x = target.x - pos.x, y = target.y - pos.y, z = target.z - pos.z })
    local right = norm3(cross(f, { x = 0, y = 0, z = 1 }))
    local down = cross(f, right)
    local r = math.rad(spec.roll_deg)
    local right2 = { x = math.cos(r) * right.x + math.sin(r) * down.x, y = math.cos(r) * right.y + math.sin(r) * down.y,
                     z = math.cos(r) * right.z + math.sin(r) * down.z }
    local down2 = { x = -math.sin(r) * right.x + math.cos(r) * down.x, y = -math.sin(r) * right.y + math.cos(r) * down.y,
                    z = -math.sin(r) * right.z + math.cos(r) * down.z }
    world.cam = { pos = pos, f = f, right = right2, up = { x = -down2.x, y = -down2.y, z = -down2.z }, fov = spec.fov_deg }
    world.tick = world.tick + 1
  end

  function host.read_camera(w, h)
    local c = world.cam
    return { pos = { x = c.pos.x * inv, y = c.pos.y * inv, z = c.pos.z * inv }, forward = c.f, right = c.right, up = c.up,
             fov_v_deg = c.fov, near = 0.1 }
  end

  function host.project(pt, w, h)
    local c = world.cam
    local d = { x = pt.x * us - c.pos.x, y = pt.y * us - c.pos.y, z = pt.z * us - c.pos.z }
    local z = dot(d, c.f)
    if z <= 0.05 then return nil end
    local f = (h / 2) / math.tan(math.rad(c.fov) / 2)
    local u, v = w / 2 + f * dot(d, c.right) / z, h / 2 - f * dot(d, c.up) / z
    if u < 0 or u >= w or v < 0 or v >= h then return nil end
    return { x = u, y = v }
  end

  function host.entities(rt)
    if not world.active then return {} end
    local out = {}
    for i = 1, #world.actors do out[i] = rt.state.handles[i] end
    return out
  end

  function host.read_bones(h) return bones_for(world.actors[h.entity_id + 1]) end

  function host.visibility(cam_pos, pt, h) return pt.z * us >= (opts.occlude_below_z or -1) end

  function host.entity_extras(h)
    local a = world.actors[h.entity_id + 1]
    local hull = {}
    for _, dx in ipairs({ -0.3, 0.3 }) do
      for _, dy in ipairs({ -0.3, 0.3 }) do
        for _, z in ipairs({ 0, 1.8 }) do
          hull[#hull + 1] = { x = (a.x + dx) * inv, y = (a.y + dy) * inv, z = z * inv }
        end
      end
    end
    return { forward = { x = math.cos(a.yaw), y = math.sin(a.yaw), z = 0 }, hull_points = hull }
  end

  function host.freeze() world.frozen = true end
  function host.unfreeze() world.frozen, world.unfreezes = false, world.unfreezes + 1 end
  function host.tick() return world.tick end

  function host.capture_image(token, w, h, rt)
    if opts.hang_capture then
      while true do rt:wait_frames(1) end
    end
    world.pending[token] = true
  end

  function host.commit_image(token, dest, rt)
    assert(world.pending[token], "no pending image for " .. token)
    world.pending[token] = nil
    local name = "staging/" .. token .. ".png"
    host.fs.write(name, opts.png or "not-a-png")
    return name
  end

  function host.discard_image(token) world.pending[token] = nil end
  function host.end_scene(rt) world.actors = {} end

  function host.sample_bones(rt)
    local a = world.actors[1] or { x = 0, y = 0, yaw = 0 }
    return { rig_id = "valve", bones = bones_for(a) }
  end

  function host.selftests(rt)
    return { { name = "freeze", ok = true, detail = "simulated" } }
  end

  return host
end
