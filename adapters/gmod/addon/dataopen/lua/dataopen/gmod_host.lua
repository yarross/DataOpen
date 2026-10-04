-- DataOpen host for Garry's Mod (CLIENT realm). Engine bindings for dataopen_rpc.lua.
--
-- Design: everything runs client-side. Characters are *clientside models* (ClientsideModel) posed by
-- activity/cycle, so there is no server/client sync problem: the pose we read is the pose we render.
-- The camera is a free camera: render.RenderView into a render target, so the game window size and the
-- player's own view do not matter. Works in singleplayer or on a listen server.
--
-- Lines marked VERIFY use a GMod API from memory and are the first suspects if `dataopen doctor` fails.

return function(R)
  local host = { name = "garrysmod", engine = "source", mod_version = "0.1.0",
                 unit_scale = 0.01905 }  -- 16 Source units = 1 foot -> meters per unit
  local us = host.unit_scale
  local cfg = { width = 1280, height = 720, fov_mode = "hor_4_3", roam_radius_m = 30, locations = {},
                hud = false, staging = "staging" }
  local world = { actors = {}, active = true, env = nil, env_on = false, center = nil, jobs = {},
                  buffers = {}, rts = {}, cam = nil, frames = 0, hooks_installed = false }
  host.world = world
  local DIR = "dataopen"

  ---------------------------------------------------------------------------------------------
  -- file system (garrysmod/data/dataopen/)
  ---------------------------------------------------------------------------------------------
  host.fs = {
    read = function(name) return file.Read(DIR .. "/" .. name, "DATA") end,
    write = function(name, data)
      file.Write(DIR .. "/" .. name, data)
      return true
    end,
  }

  local function write_binary(name, data)
    local f = file.Open(DIR .. "/" .. name, "wb", "DATA")
    if not f then error("cannot open " .. name .. " for writing in garrysmod/data/" .. DIR) end
    f:Write(data)
    f:Close()
  end

  function host.now() return SysTime() end
  function host.tick() return FrameNumber() end
  function host.game_version() return tostring(VERSION) end
  function host.render_size() return cfg.width, cfg.height end

  ---------------------------------------------------------------------------------------------
  -- randomization vocabulary (what this game can vary)
  ---------------------------------------------------------------------------------------------
  local WEAPONS = {
    { name = "none", model = nil, hold = "none" },
    { name = "pistol", model = "models/weapons/w_pistol.mdl", hold = "pistol" },
    { name = "smg", model = "models/weapons/w_smg1.mdl", hold = "smg1" },
    { name = "ar2", model = "models/weapons/w_irifle.mdl", hold = "ar2" },
    { name = "shotgun", model = "models/weapons/w_shotgun.mdl", hold = "shotgun" },
    { name = "crowbar", model = "models/weapons/w_crowbar.mdl", hold = "melee" },
    { name = "crossbow", model = "models/weapons/w_crossbow.mdl", hold = "crossbow" },
    { name = "rpg", model = "models/weapons/w_rocket_launcher.mdl", hold = "rpg" },
  }
  local WEAPON_BY_NAME = {}
  local weapon_names = {}
  for _, w in ipairs(WEAPONS) do
    WEAPON_BY_NAME[w.name] = w
    weapon_names[#weapon_names + 1] = w.name
  end
  local STANCE_ACT = { stand = "IDLE", walk = "WALK", run = "RUN", crouch = "IDLE_CROUCH", crouch_walk = "WALK_CROUCH" }

  local function model_names()
    local names = {}
    for name in pairs(player_manager.AllValidModels()) do names[#names + 1] = name end  -- VERIFY
    table.sort(names)
    if #names == 0 then names = { "kleiner" } end
    return names
  end

  function host.parameter_space()
    local models = R.json.array(model_names())
    return {
      environment = {
        weather = { type = "categorical", choices = R.json.array({ "clear", "overcast", "fog" }),
                    weights = R.json.array({ 0.5, 0.3, 0.2 }) },
        saturation = { type = "uniform", lo = 0.6, hi = 1.2 },
        contrast = { type = "uniform", lo = 0.85, hi = 1.2 },
      },
      actor = {
        rig = { type = "constant", value = "auto" },
        model = { type = "categorical", choices = models },
        weapon = { type = "categorical", choices = R.json.array(weapon_names),
                   weights = R.json.array({ 0.35, 0.1, 0.1, 0.15, 0.1, 0.1, 0.05, 0.05 }) },
        skin = { type = "uniform", lo = 0, hi = 1 },
        bodygroup_seed = { type = "uniform", lo = 0, hi = 1 },
      },
      actor_frame = {
        stance = { type = "categorical", choices = R.json.array({ "stand", "walk", "run", "crouch", "crouch_walk" }),
                   weights = R.json.array({ 0.35, 0.25, 0.15, 0.15, 0.10 }) },
        phase = { type = "uniform", lo = 0, hi = 1 },
        yaw = { type = "uniform", lo = -180, hi = 180 },
        aim_pitch = { type = "uniform", lo = -30, hi = 30 },
        aim_yaw = { type = "uniform", lo = -30, hi = 30 },
      },
    }
  end

  function host.rigs() return { "valvebiped" } end
  function host.bone_map(rig) return rig == "valvebiped" and R.maps.valvebiped or {} end

  ---------------------------------------------------------------------------------------------
  -- lifecycle
  ---------------------------------------------------------------------------------------------
  local function install_hooks()
    if world.hooks_installed then return end
    world.hooks_installed = true

    -- All rendering and ToScreen work runs here: GMod only allows it inside render hooks.
    hook.Add("PostRender", "dataopen_jobs", function()
      local job = table.remove(world.jobs, 1)
      if not job then return end
      local ok, err = pcall(job.run, job)
      if not ok then job.error = tostring(err) end
      job.done = true
    end)

    hook.Add("RenderScreenspaceEffects", "dataopen_grade", function()
      if not world.env_on or not world.env then return end
      local e = world.env
      local t = tonumber(e.time_of_day) or 12
      local day = math.max(0, math.sin(math.pi * (t - 6) / 12))
      local dusk = math.max(0, 1 - math.abs(day - 0.2) / 0.2)
      local sat = tonumber(e.saturation) or 1
      local con = tonumber(e.contrast) or 1
      if e.weather == "overcast" then sat, con = sat * 0.85, con * 0.92 end
      DrawColorModify({
        ["$pp_colour_addr"] = 0.06 * dusk, ["$pp_colour_addg"] = 0.02 * dusk, ["$pp_colour_addb"] = 0.05 * (1 - day),
        ["$pp_colour_brightness"] = -0.30 * (1 - day) + 0.08 * (tonumber(e.exposure_ev) or 0),
        ["$pp_colour_contrast"] = con, ["$pp_colour_colour"] = sat,
        ["$pp_colour_mulr"] = 0, ["$pp_colour_mulg"] = 0, ["$pp_colour_mulb"] = 0,
      })
    end)

    local function fog(scale)
      if not world.env_on or not world.env then return end
      local d = tonumber(world.env.fog_density) or 0
      if d <= 0 then return end
      local far = (3.0 / d) / us * (scale or 1)  -- meteorological visibility ~ 3/density
      render.FogMode(MATERIAL_FOG_LINEAR)
      render.FogStart(far * 0.05)
      render.FogEnd(far)
      render.FogMaxDensity(0.95)
      render.FogColor(170, 178, 190)
      return true
    end
    hook.Add("SetupWorldFog", "dataopen_fog", function() return fog(1) end)
    hook.Add("SetupSkyboxFog", "dataopen_skyfog", function(scale) return fog(1 / (scale or 1)) end)
  end

  function host.init(options, rt, image)
    for k, v in pairs(options or {}) do cfg[k] = v end
    if image and (image.width or 0) > 0 then cfg.width, cfg.height = image.width, image.height end
    file.CreateDir(DIR)
    file.CreateDir(DIR .. "/" .. cfg.staging)
    cfg.locations = {}
    local raw = file.Read(DIR .. "/locations.json", "DATA")
    if raw then
      local t = util.JSONToTable(raw)
      for _, p in ipairs(t or {}) do cfg.locations[#cfg.locations + 1] = Vector(p[1], p[2], p[3]) end
    end
    install_hooks()
  end

  ---------------------------------------------------------------------------------------------
  -- scene: environment + actors
  ---------------------------------------------------------------------------------------------
  local function lcg(seed)
    local s = seed % 2147483647
    if s <= 0 then s = s + 2147483646 end
    return function()
      s = (s * 48271) % 2147483647
      return s / 2147483647
    end
  end

  local function ground_at(x, y, z0)
    local tr = util.TraceLine({ start = Vector(x, y, z0 + 512), endpos = Vector(x, y, z0 - 512),
                                mask = MASK_SOLID_BRUSHONLY })
    if not tr.Hit or tr.StartSolid or tr.HitNormal.z < 0.9 then return nil end
    local hull = util.TraceHull({ start = tr.HitPos + Vector(0, 0, 4), endpos = tr.HitPos + Vector(0, 0, 4),
                                  mins = Vector(-16, -16, 0), maxs = Vector(16, 16, 72), mask = MASK_SOLID })
    if hull.Hit or hull.StartSolid then return nil end
    return tr.HitPos
  end

  local function pick_center(rnd, scene)
    local base = LocalPlayer():GetPos()
    if #cfg.locations > 0 then base = cfg.locations[(scene.scene_index % #cfg.locations) + 1] end
    local radius = (#cfg.locations > 0 and 3 or cfg.roam_radius_m) / us
    for _ = 1, 40 do
      local r, th = radius * math.sqrt(rnd()), rnd() * 2 * math.pi
      local g = ground_at(base.x + r * math.cos(th), base.y + r * math.sin(th), base.z)
      if g then return g end
    end
    return nil
  end

  local function clear_actors()
    for _, a in ipairs(world.actors) do
      if IsValid(a.ent) then a.ent:Remove() end
      if a.wep and IsValid(a.wep) then a.wep:Remove() end
    end
    world.actors = {}
  end

  local function apply_look(ent, params)
    local nskin = ent:SkinCount()  -- VERIFY
    if nskin and nskin > 1 then ent:SetSkin(math.min(nskin - 1, math.floor((params.skin or 0) * nskin))) end
    local u = tonumber(params.bodygroup_seed) or 0
    for i = 0, (ent:GetNumBodyGroups() or 0) - 1 do
      local n = ent:GetBodygroupCount(i)
      if n and n > 1 then
        u = (u * 7919 + 0.137 * (i + 1)) % 1
        ent:SetBodygroup(i, math.min(n - 1, math.floor(u * n)))
      end
    end
  end

  function host.begin_scene(scene, rt)
    clear_actors()
    world.env = scene.environment
    local rnd = lcg(scene.seed)
    world.center = pick_center(rnd, scene)
    if not world.center then error("no flat, unobstructed ground found near the player: move the player to an open area") end
    local radius = math.min(scene.area.radius, 12) / us
    local sep = (scene.area.min_separation or 1.0) / us
    local handles = {}
    for i, params in ipairs(scene.actors) do
      local pos
      for _ = 1, 30 do
        local r, th = radius * math.sqrt(rnd()), rnd() * 2 * math.pi
        local cand = ground_at(world.center.x + r * math.cos(th), world.center.y + r * math.sin(th), world.center.z)
        if cand then
          local ok = true
          for _, a in ipairs(world.actors) do if a.pos:Distance(cand) < sep then ok = false break end end
          if ok then pos = cand break end
        end
      end
      if pos then
        local path = player_manager.TranslatePlayerModel(params.model)  -- VERIFY
        local ent = ClientsideModel(path, RENDERGROUP_OPAQUE)
        if IsValid(ent) then
          ent:SetPos(pos)
          ent:SetAngles(Angle(0, rnd() * 360, 0))
          ent:SetNoDraw(false)
          apply_look(ent, params)
          local weapon = WEAPON_BY_NAME[params.weapon] or WEAPON_BY_NAME.none
          local wep
          if weapon.model then
            wep = ClientsideModel(weapon.model, RENDERGROUP_OPAQUE)
            if IsValid(wep) then
              wep:SetPos(pos)
              wep:SetParent(ent)
              wep:AddEffects(EF_BONEMERGE)  -- VERIFY: weapons follow the hand via bonemerge
            end
          end
          local rig = ent:LookupBone("ValveBiped.Bip01_Pelvis") and "valvebiped" or "generic"
          world.actors[#world.actors + 1] = { ent = ent, wep = wep, pos = pos, yaw = rnd() * 360, hold = weapon.hold,
                                              rig = rig, model = params.model, weapon = weapon.name }
          handles[#handles + 1] = { entity_id = #world.actors - 1, rig_id = rig,
                                    meta = { model = params.model, weapon = weapon.name } }
        end
      end
    end
    if #handles == 0 then error("could not place any actor (blocked ground?)") end
    world.active = true
    return handles
  end

  local function set_pose_param(ent, name, value)
    ent:SetPoseParameter(name, value)  -- VERIFY: player models use aim_pitch/aim_yaw (-45..45)
  end

  local function resolve_seq(ent, stance, hold)
    local base = STANCE_ACT[stance] or "IDLE"
    local names = {}
    if hold and hold ~= "none" then names[#names + 1] = "ACT_HL2MP_" .. base .. "_" .. string.upper(hold) end
    names[#names + 1] = "ACT_HL2MP_" .. base
    for _, n in ipairs(names) do
      local act = _G[n]
      if act then
        local seq = ent:SelectWeightedSequence(act)
        if seq and seq >= 0 then return seq end
      end
    end
    return 0
  end

  function host.apply_frame(frame, handles, rt)
    for i, a in ipairs(world.actors) do
      local f = frame.actor_frame[i] or {}
      local ent = a.ent
      if IsValid(ent) then
        ent:ResetSequence(resolve_seq(ent, f.stance or "stand", a.hold))  -- VERIFY
        ent:SetPlaybackRate(0)
        ent:SetCycle(math.max(0, math.min(0.999, f.phase or 0)))
        ent:SetAngles(Angle(0, f.yaw or a.yaw, 0))
        set_pose_param(ent, "aim_pitch", f.aim_pitch or 0)
        set_pose_param(ent, "aim_yaw", f.aim_yaw or 0)
        ent:InvalidateBoneCache()
        ent:SetupBones()
      end
    end
  end

  function host.set_active(handles, active)
    world.active = active
    for _, a in ipairs(world.actors) do
      if IsValid(a.ent) then a.ent:SetNoDraw(not active) end
      if a.wep and IsValid(a.wep) then a.wep:SetNoDraw(not active) end
    end
  end

  function host.entities(rt)
    if not world.active then return {} end
    return rt.state.handles
  end

  function host.end_scene(rt)
    clear_actors()
    world.env_on = false
  end

  ---------------------------------------------------------------------------------------------
  -- camera (free camera, rendered with render.RenderView)
  ---------------------------------------------------------------------------------------------
  local function engine_fov(fov_v, w, h)
    local aspect = w / h
    if cfg.fov_mode == "vertical" then return fov_v end
    local hfov = 2 * math.deg(math.atan(math.tan(math.rad(fov_v) / 2) * aspect))
    if cfg.fov_mode == "horizontal" then return hfov end
    -- "hor_4_3": Source's hor+ convention: the value is the horizontal FOV *at 4:3* (VERIFY for RenderView)
    return 2 * math.deg(math.atan(math.tan(math.rad(fov_v) / 2) * (4 / 3)))
  end

  function host.place_camera(spec, handles, w, h, rt)
    local tgt = world.actors[(spec.target_index or 0) + 1] or world.actors[1]
    local target = (tgt and tgt.pos or world.center) + Vector(0, 0, 56)
    local y, p = math.rad(spec.yaw_deg), math.rad(spec.pitch_deg)
    local dist = spec.distance / us
    local pos = target + Vector(math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)) * dist
    pos.z = pos.z + spec.height_offset / us
    local tr = util.TraceLine({ start = target, endpos = pos, mask = MASK_SOLID_BRUSHONLY })
    if tr.Hit then pos = tr.HitPos + tr.HitNormal * 6 end  -- never place the camera inside a wall
    local ang = (target - pos):Angle()
    ang.r = spec.roll_deg
    world.cam = { origin = pos, angles = ang, fov_v = spec.fov_deg, fov = engine_fov(spec.fov_deg, w, h) }
  end

  function host.read_camera(w, h)
    local c = world.cam
    return { pos = c.origin, forward = c.angles:Forward(), right = c.angles:Right(), up = c.angles:Up(),
             fov_v_deg = c.fov_v, near = 4 * us }
  end

  function host.freeze() world.frozen = true end        -- poses are static: nothing to pause
  function host.unfreeze() world.frozen = false end

  ---------------------------------------------------------------------------------------------
  -- rendering jobs (run inside PostRender)
  ---------------------------------------------------------------------------------------------
  local function get_rt(w, h)
    local key = w .. "x" .. h
    if not world.rts[key] then
      world.rts[key] = GetRenderTargetEx("dataopen_" .. key, w, h, RT_SIZE_LITERAL, MATERIAL_RT_DEPTH_SEPARATE,
                                         bit.bor(1, 256), 0, IMAGE_FORMAT_RGB888)  -- VERIFY flags/format
    end
    return world.rts[key]
  end

  local function in_rt(w, h, fn)
    render.PushRenderTarget(get_rt(w, h))
    local ok, a = pcall(fn)
    render.PopRenderTarget()
    if not ok then error(a, 0) end
    return a
  end

  local function submit(rt, run, timeout)
    local job = { run = run, done = false }
    world.jobs[#world.jobs + 1] = job
    if not rt:wait_until(function() return job.done end, timeout or 15) then
      error("the render hook did not run (is the game window focused and not paused or in the menu?)")
    end
    if job.error then error(job.error, 0) end
    return job
  end

  function host.project_batch(points, w, h, rt)
    local c = world.cam
    local out = {}
    submit(rt, function()
      in_rt(w, h, function()
        cam.Start3D(c.origin, c.angles, c.fov, 0, 0, w, h)
        for i, pt in ipairs(points) do
          local sp = Vector(pt.x, pt.y, pt.z):ToScreen()  -- VERIFY: inside cam.Start3D it projects into that view
          if sp.visible and sp.x >= 0 and sp.x < w and sp.y >= 0 and sp.y < h then
            out[i] = { x = sp.x, y = sp.y }
          else
            out[i] = false
          end
        end
        cam.End3D()
      end)
    end)
    return out
  end

  function host.capture_image(token, w, h, rt)
    local c = world.cam
    local job = submit(rt, function(self)
      in_rt(w, h, function()
        render.Clear(0, 0, 0, 255, true, true)
        world.env_on = true
        render.RenderView({ origin = c.origin, angles = c.angles, x = 0, y = 0, w = w, h = h, fov = c.fov,
                            aspectratio = w / h, drawviewmodel = false, drawhud = false, dopostprocess = true,
                            drawmonitors = true, znear = 4 })
        world.env_on = false
        self.data = render.Capture({ format = "png", x = 0, y = 0, w = w, h = h, quality = 100 })  -- VERIFY
      end)
    end)
    if not job.data or #job.data == 0 then error("render.Capture returned no data") end
    world.buffers[token] = job.data
  end

  function host.commit_image(token, dest, rt)
    local data = world.buffers[token]
    if not data then error("no pending image for " .. tostring(token)) end
    world.buffers[token] = nil
    local rel = cfg.staging .. "/" .. token .. ".dat"  -- .dat is always allowed in data/; the core renames it
    write_binary(rel, data)
    return rel
  end

  function host.discard_image(token) world.buffers[token] = nil end

  ---------------------------------------------------------------------------------------------
  -- per-entity info
  ---------------------------------------------------------------------------------------------
  function host.read_bones(h)
    local a = world.actors[h.entity_id + 1]
    local ent = a.ent
    ent:SetupBones()
    local bones = {}
    for i = 0, ent:GetBoneCount() - 1 do
      local name = ent:GetBoneName(i)
      if name and name ~= "__INVALIDBONE__" then
        local pos = ent:GetBonePosition(i)
        if not pos or (pos.x == 0 and pos.y == 0 and pos.z == 0) then
          local m = ent:GetBoneMatrix(i)
          pos = m and m:GetTranslation() or nil
        end
        if pos then bones[name] = pos end
      end
    end
    return bones
  end

  function host.visibility(cam_pos, pt, h)
    local tr = util.TraceLine({ start = cam_pos, endpos = Vector(pt.x, pt.y, pt.z), mask = MASK_OPAQUE })
    return (not tr.Hit) or tr.Fraction > 0.97
  end

  function host.entity_extras(h)
    local ent = world.actors[h.entity_id + 1].ent
    local mins, maxs = ent:GetRenderBounds()
    local hull = {}
    for _, x in ipairs({ mins.x, maxs.x }) do
      for _, y in ipairs({ mins.y, maxs.y }) do
        for _, z in ipairs({ mins.z, maxs.z }) do hull[#hull + 1] = ent:LocalToWorld(Vector(x, y, z)) end
      end
    end
    return { forward = ent:GetForward(), hull_points = hull }
  end

  ---------------------------------------------------------------------------------------------
  -- self tests (shown by `dataopen doctor`)
  ---------------------------------------------------------------------------------------------
  function host.sample_bones(rt)
    local a = world.actors[1]
    if a and IsValid(a.ent) then return { rig_id = a.rig, bones = host.read_bones({ entity_id = 0 }) } end
    local tmp = ClientsideModel(player_manager.TranslatePlayerModel(model_names()[1]), RENDERGROUP_OPAQUE)
    if not IsValid(tmp) then return nil end
    tmp:SetPos(LocalPlayer():GetPos())
    tmp:SetupBones()
    local save = world.actors
    world.actors = { { ent = tmp } }
    local bones = host.read_bones({ entity_id = 0 })
    world.actors = save
    local rig = tmp:LookupBone("ValveBiped.Bip01_Pelvis") and "valvebiped" or "generic"
    tmp:Remove()
    return { rig_id = rig, bones = bones }
  end

  function host.selftests(rt)
    local checks = {}
    local function add(name, ok, detail, hint) checks[#checks + 1] = { name = name, ok = ok, detail = detail, hint = hint } end
    add("player_models", #model_names() > 0, #model_names() .. " player models available")
    local ok_file = pcall(function() write_binary(cfg.staging .. "/selftest.dat", "x") end)
    add("data_folder_write", ok_file, "garrysmod/data/" .. DIR .. " is writable",
        "the game cannot write its data folder: check permissions / the mailbox path")
    local ok_ground = pick_center(lcg(1), { scene_index = 0 }) ~= nil
    add("ground", ok_ground, ok_ground and "flat ground found near the player" or "no flat ground near the player",
        "stand in an open flat area (gm_flatgrass is ideal) before running the doctor")
    local ok_job, err = pcall(function() host.project_batch({}, 64, 64, rt) end)
    local c0 = world.cam
    if not c0 then
      world.cam = { origin = Vector(0, 0, 0), angles = Angle(0, 0, 0), fov = 90, fov_v = 90 }
      ok_job, err = pcall(function() host.project_batch({}, 64, 64, rt) end)
      world.cam = nil
    end
    add("render_hook", ok_job, ok_job and "PostRender jobs run" or tostring(err),
        "keep the game window focused and out of the pause menu while collecting")
    return checks
  end

  return host
end
