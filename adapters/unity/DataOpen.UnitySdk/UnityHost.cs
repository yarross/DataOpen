// Protocol handlers (docs/PROTOCOL.md) for Unity games. Handlers are coroutines (IEnumerator), so they can
// wait for frames exactly like the Lua runtime's handlers do.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace DataOpen
{
    public sealed class Call
    {
        public Dictionary<string, object> Req;
        public Dictionary<string, object> Params;
        public Dictionary<string, object> Result = new Dictionary<string, object>();
        public string ErrorType;
        public string ErrorMessage;
        public bool Failed { get { return ErrorType != null; } }
        public void Fail(string type, string message) { ErrorType = type; ErrorMessage = message; }
    }

    public sealed class UnityHost
    {
        public const int Protocol = 1;
        readonly IGameBindings bindings;
        readonly string mailboxDir;
        readonly Action<string> log;
        readonly FreeCam freeCam = new FreeCam();
        readonly Dictionary<string, Texture2D> pending = new Dictionary<string, Texture2D>();
        List<string> keypoints = new List<string>();
        BoneResolver resolver;
        Dictionary<string, object> options = new Dictionary<string, object>();
        int width = 1280, height = 720, frames;

        public UnityHost(IGameBindings bindings, string mailboxDir, Action<string> log)
        {
            this.bindings = bindings;
            this.mailboxDir = mailboxDir;
            this.log = log;
        }

        public IEnumerator Handle(Call c)
        {
            switch (Json.Str(c.Req, "method"))
            {
                case "hello": return Hello(c);
                case "begin_scene": return BeginScene(c);
                case "capture_frame": return CaptureFrame(c);
                case "release": return Done();
                case "commit": return Commit(c);
                case "discard": return Discard(c);
                case "end_scene": return EndScene(c);
                case "health": return Health(c);
                case "selftest": return SelfTest(c);
                case "shutdown": return Done();
                default:
                    c.Fail("ProtocolError", "unknown method " + Json.Str(c.Req, "method"));
                    return Done();
            }
        }

        static IEnumerator Done() { yield break; }

        // ---------------------------------------------------------------------------------------------
        IEnumerator Hello(Call c)
        {
            options = Json.Obj(c.Params, "options") ?? new Dictionary<string, object>();
            keypoints = new List<string>();
            var schema = Json.Obj(c.Params, "schema");
            var kps = Json.Arr(schema, "keypoints");
            if (kps != null) foreach (object k in kps) keypoints.Add(Convert.ToString(k));
            resolver = new BoneResolver(keypoints, Json.Obj(c.Params, "bone_map"));
            var img = Json.Obj(c.Params, "image");
            if (Json.Num(img, "width") > 0) { width = (int)Json.Num(img, "width"); height = (int)Json.Num(img, "height"); }
            Humanoids.ClearCache();
            bindings.Init(options, log);
            c.Result["protocol"] = Protocol;
            c.Result["game"] = bindings.GameName;
            c.Result["engine"] = "unity";
            c.Result["game_version"] = bindings.GameVersion();
            c.Result["mod_version"] = "0.1.0";
            c.Result["capabilities"] = new List<object> { "probes", "engine_visibility", "hull_points", "image_engine" };
            c.Result["image"] = new Dictionary<string, object> { { "width", width }, { "height", height } };
            c.Result["schema_errors"] = new List<object>();
            c.Result["parameter_space"] = bindings.ParameterSpace();
            yield break;
        }

        IEnumerator BeginScene(Call c)
        {
            IEnumerator inner = bindings.BeginScene(Json.Obj(c.Params, "scene"));
            while (inner.MoveNext()) yield return inner.Current;
            // Observe mode: the population is whatever the game has; the core learns actors from each frame.
            c.Result["handles"] = new List<object>();
        }

        IEnumerator EndScene(Call c)
        {
            bindings.EndScene();
            foreach (var t in pending.Values) if (t != null) UnityEngine.Object.Destroy(t);
            pending.Clear();
            yield break;
        }

        IEnumerator Health(Call c)
        {
            c.Result["ok"] = true;
            c.Result["frames"] = frames;
            yield break;
        }

        // ---------------------------------------------------------------------------------------------
        IEnumerator CaptureFrame(Call c)
        {
            var p = c.Params;
            int w = (int)Json.Num(p, "width", width), h = (int)Json.Num(p, "height", height);
            string token = Json.Str(p, "frame_id");
            var spec = Json.Obj(Json.Obj(p, "frame"), "camera");
            float interval = (float)Json.Num(options, "frame_interval_s", 0.25);
            int settle = (int)Json.Num(options, "settle_ticks", 1);
            float radius = (float)Json.Num(options, "observe_radius_m", 40.0);

            float until = Time.realtimeSinceStartup + interval;  // let the world move between frames
            while (Time.realtimeSinceStartup < until) yield return null;
            for (int i = 0; i < settle; i++) yield return null;
            yield return new WaitForEndOfFrame();                // bones, camera and pixels all from this frame

            Vector3 anchor = bindings.Anchor();
            // Observe mode cannot hide anyone: on negative frames we still report whoever is around, and the core's
            // validators reject the frame if a person is visible.
            List<Actor> actors = Humanoids.Discover(anchor, radius, resolver);

            // ---- camera placement relative to the target actor (same spec as the other adapters) ----
            Actor target = actors.Count > 0 ? actors[((int)Json.Num(spec, "target_index")) % actors.Count] : null;
            var skeletons = new List<ResolvedSkeleton>();
            foreach (Actor a in actors) skeletons.Add(resolver.Resolve(a));
            Vector3 aim = anchor;
            if (target != null)
            {
                ResolvedSkeleton ts = skeletons[actors.IndexOf(target)];
                int pi = keypoints.IndexOf("pelvis");
                aim = (pi >= 0 && ts.Valid[pi]) ? ts.Positions[pi] : target.Root.transform.position + Vector3.up;
            }
            float dist = (float)Json.Num(spec, "distance", 6.0);
            float yaw = (float)(Json.Num(spec, "yaw_deg") * Math.PI / 180.0);
            float pitch = (float)(Json.Num(spec, "pitch_deg") * Math.PI / 180.0);
            Vector3 dir = new Vector3(Mathf.Cos(pitch) * Mathf.Cos(yaw), Mathf.Sin(pitch), Mathf.Cos(pitch) * Mathf.Sin(yaw));
            Vector3 pos = aim + dir * dist + Vector3.up * (float)Json.Num(spec, "height_offset");
            float blocked = FreeCam.FirstBlocker(aim, pos, target != null ? target.Root.transform : null);
            if (blocked >= 0f) pos = aim + (pos - aim).normalized * Mathf.Max(0.5f, blocked - 0.15f);  // never inside a wall

            freeCam.Ensure(w, h);
            freeCam.Place(pos, aim, (float)Json.Num(spec, "roll_deg"), (float)Json.Num(spec, "fov_deg", 60.0));
            Camera cam = freeCam.Cam;
            Transform ct = cam.transform;

            // ---- entities ----
            var entities = new List<object>();
            var warnings = new List<object>();
            for (int i = 0; i < actors.Count; i++)
            {
                Actor a = actors[i];
                ResolvedSkeleton sk = skeletons[i];
                var flat = new List<object>();
                var valid = new List<object>();
                var vis = new List<object>();
                for (int k = 0; k < keypoints.Count; k++)
                {
                    Vector3 q = sk.Positions[k];
                    flat.Add(q.x); flat.Add(q.y); flat.Add(q.z);
                    valid.Add(sk.Valid[k]);
                    bool occluded = false;
                    if (sk.Valid[k])
                    {
                        float hit = FreeCam.FirstBlocker(ct.position, q, a.Root.transform);
                        occluded = hit >= 0f && hit < (q - ct.position).magnitude - 0.05f;
                    }
                    vis.Add(occluded ? 1 : 2);
                }
                var e = new Dictionary<string, object>
                {
                    { "entity_id", a.Id }, { "rig_id", a.RigId }, { "skeleton_world", flat }, { "joint_valid", valid },
                    { "engine_visibility", vis }, { "meta", new Dictionary<string, object>() },
                };
                if (a.Anim != null && a.Anim.isHuman)  // humanoid rigs face +Z: an independent facing for the L/R check
                {
                    Vector3 f = a.Root.transform.forward;
                    ((Dictionary<string, object>)e["meta"])["forward"] = new List<object> { f.x, f.y, f.z };
                }
                var hull = new List<object>();
                foreach (Vector3 q in Humanoids.HullPoints(a)) { hull.Add(q.x); hull.Add(q.y); hull.Add(q.z); }
                if (hull.Count > 0) e["hull_points"] = hull;
                if (sk.Unmapped.Count > 0) warnings.Add("entity " + a.Id + ": unmapped keypoints " + string.Join(",", sk.Unmapped.ToArray()));
                entities.Add(e);
            }

            // ---- pixels (held until commit/discard) ----
            if (Json.Str(p, "image_mode", "engine") == "engine")
            {
                Texture2D old;
                if (pending.TryGetValue(token, out old) && old != null) UnityEngine.Object.Destroy(old);
                pending[token] = freeCam.Render();
            }

            // ---- probes: the engine's own projection of known points ----
            var probes = new List<object>();
            var probePts = new List<Vector3> { ct.position + ct.forward * 6 + ct.right * 1.5f,
                                               ct.position + ct.forward * 12 - ct.right * 3 + ct.up * 2,
                                               ct.position + ct.forward * 4 + ct.up * 1.2f };
            int hi = keypoints.IndexOf("head") >= 0 ? keypoints.IndexOf("head") : 0;
            for (int i = 0; i < actors.Count && i < 3; i++) probePts.Add(skeletons[i].Positions[hi]);
            foreach (Vector3 pt in probePts)
            {
                float u, v;
                var pr = new Dictionary<string, object> { { "world", new List<object> { pt.x, pt.y, pt.z } } };
                if (freeCam.Project(pt, out u, out v)) pr["screen"] = new List<object> { u, v };
                probes.Add(pr);
            }

            frames++;
            c.Result["frame_token"] = token;
            c.Result["tick"] = Time.frameCount;
            c.Result["camera"] = new Dictionary<string, object>
            {
                { "width", w }, { "height", h }, { "near", cam.nearClipPlane },
                { "pos", new List<object> { ct.position.x, ct.position.y, ct.position.z } },
                { "forward", new List<object> { ct.forward.x, ct.forward.y, ct.forward.z } },
                { "right", new List<object> { ct.right.x, ct.right.y, ct.right.z } },
                { "up", new List<object> { ct.up.x, ct.up.y, ct.up.z } },
                { "fov_v_deg", cam.fieldOfView },
            };
            c.Result["entities"] = entities;
            c.Result["probes"] = probes;
            c.Result["warnings"] = warnings;
        }

        IEnumerator Commit(Call c)
        {
            string token = Json.Str(c.Params, "frame_token");
            string dest = Json.Str(c.Params, "dest");
            Texture2D tex;
            if (!pending.TryGetValue(token, out tex) || tex == null) { c.Fail("NoSuchFrame", "no pending image for " + token); yield break; }
            pending.Remove(token);
            Directory.CreateDirectory(Path.GetDirectoryName(dest));
            string ext = Path.GetExtension(dest).ToLowerInvariant();
            byte[] bytes = (ext == ".jpg" || ext == ".jpeg") ? ImageConversion.EncodeToJPG(tex, 95) : ImageConversion.EncodeToPNG(tex);
            File.WriteAllBytes(dest, bytes);
            UnityEngine.Object.Destroy(tex);
        }

        IEnumerator Discard(Call c)
        {
            Texture2D tex;
            string token = Json.Str(c.Params, "frame_token");
            if (pending.TryGetValue(token, out tex) && tex != null) UnityEngine.Object.Destroy(tex);
            pending.Remove(token);
            yield break;
        }

        IEnumerator SelfTest(Call c)
        {
            var checks = new List<object>();
            Action<string, bool, string, string, Dictionary<string, object>> add = (name, ok, detail, hint, data) =>
            {
                var d = new Dictionary<string, object> { { "name", name }, { "ok", ok }, { "detail", detail } };
                if (!string.IsNullOrEmpty(hint)) d["hint"] = hint;
                if (data != null) d["data"] = data;
                checks.Add(d);
            };

            Vector3 anchor = bindings.Anchor();
            List<Actor> actors = Humanoids.Discover(anchor, (float)Json.Num(options, "observe_radius_m", 40.0), resolver);
            if (actors.Count == 0)
            {
                add("bone_mapping", false, "no humanoid found within " + Json.Num(options, "observe_radius_m", 40.0) + " m",
                    "stand near NPCs/players (or let the bindings spawn some); a humanoid needs an Animator with ~10 mappable bones", null);
            }
            else
            {
                ResolvedSkeleton sk = resolver.Resolve(actors[0]);
                var unmapped = new List<object>();
                foreach (string u in sk.Unmapped) unmapped.Add(u);
                var names = new List<object>();
                foreach (string n in actors[0].RawNames) { if (names.Count >= 200) break; names.Add(n); }
                add("bone_mapping", sk.Unmapped.Count == 0, (keypoints.Count - sk.Unmapped.Count) + "/" + keypoints.Count +
                    " keypoints resolved on '" + actors[0].Root.name + "' (" + actors[0].RigId + ")",
                    sk.Unmapped.Count > 0 ? "add overrides to the profile's [bones] table using the names in bones_found" : null,
                    new Dictionary<string, object> { { "unmapped", unmapped }, { "bones_found", names } });
            }
            add("main_camera", Camera.main != null, Camera.main != null ? "Camera.main found (render settings are copied from it)" : "no Camera.main",
                Camera.main == null ? "the free camera will use default settings; the picture may differ from the game's look" : null, null);
            try
            {
                freeCam.Ensure(32, 32);
                freeCam.Place(anchor + Vector3.up * 2, anchor, 0f, 60f);
                Texture2D t = freeCam.Render();
                UnityEngine.Object.Destroy(t);
                add("render", true, "free camera rendered a test image", null, null);
            }
            catch (Exception e) { add("render", false, e.Message, "Camera.Render failed: check the render pipeline (URP/HDRP need a camera copy)", null); }
            try
            {
                string probe = Path.Combine(mailboxDir, "selftest.tmp");
                File.WriteAllText(probe, "x");
                File.Delete(probe);
                add("mailbox_write", true, mailboxDir + " is writable", null, null);
            }
            catch (Exception e) { add("mailbox_write", false, e.Message, "the game process cannot write the mailbox folder", null); }
            foreach (SelfTestResult r in bindings.SelfTests()) add(r.Name, r.Ok, r.Detail, r.Hint, null);
            c.Result["checks"] = checks;
            yield break;
        }
    }
}
