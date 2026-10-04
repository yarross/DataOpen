// Valheim-specific control (time of day, weather, optional enemy spawning, ghost mode so enemies ignore the player).
// Every Valheim member is reached by NAME through reflection (VERIFY): a renamed member does not break the build,
// it shows up as a failed self-test in `dataopen doctor`.
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DataOpen.Valheim
{
    public sealed class ValheimBindings : GenericBindings
    {
        static readonly Dictionary<string, string> Weather = new Dictionary<string, string>
        {
            { "clear", "Clear" }, { "overcast", "Misty" }, { "rain", "Rain" }, { "fog", "DeepForest Mist" }, { "snow", "Snow" },
        };
        static readonly string[] Prefabs = { "Skeleton", "Draugr", "Goblin", "GoblinArcher", "GoblinBrute", "Draugr_Ranged" };

        readonly List<GameObject> spawned = new List<GameObject>();
        readonly List<string> notes = new List<string>();

        public override string GameName { get { return "Valheim"; } }

        public override string GameVersion()
        {
            try
            {
                object v = Reflect.CallStatic(Reflect.FindType("Version"), "GetVersionString");
                if (v != null) return v.ToString();
            }
            catch (Exception) { }
            return Application.version;
        }

        public override Dictionary<string, object> ParameterSpace()
        {
            var weather = new Dictionary<string, object>
            {
                { "type", "categorical" },
                { "choices", new List<object> { "clear", "overcast", "rain", "fog", "snow" } },
                { "weights", new List<object> { 0.4, 0.2, 0.2, 0.1, 0.1 } },
            };
            var prefab = new Dictionary<string, object>
            {
                { "type", "categorical" }, { "choices", new List<object>(Prefabs) },
            };
            return new Dictionary<string, object>
            {
                { "environment", new Dictionary<string, object> { { "weather", weather } } },
                { "actor", new Dictionary<string, object> { { "prefab", prefab } } },
                { "actor_frame", new Dictionary<string, object>() },
            };
        }

        object LocalPlayer() { return Reflect.StaticMember(Reflect.FindType("Player"), "m_localPlayer"); }

        public override Vector3 Anchor()
        {
            var p = LocalPlayer() as Component;
            return p != null ? p.transform.position : base.Anchor();
        }

        void ApplyEnvironment(Dictionary<string, object> env)
        {
            notes.Clear();
            object envMan = Reflect.StaticMember(Reflect.FindType("EnvMan"), "instance");
            if (envMan == null) { notes.Add("EnvMan.instance not found"); return; }
            double tod = Json.Num(env, "time_of_day", 12.0) % 24.0;
            bool t1 = Reflect.SetMember(envMan, "m_debugTimeOfDay", true);
            bool t2 = Reflect.SetMember(envMan, "m_debugTime", (float)(tod / 24.0));
            if (!t1 || !t2) notes.Add("time of day: EnvMan.m_debugTimeOfDay/m_debugTime not found");
            string name;
            if (Weather.TryGetValue(Json.Str(env, "weather", "clear"), out name))
            {
                try { Reflect.Call(envMan, "SetDebugEnv", name, true); }
                catch (Exception e) { notes.Add("weather: SetDebugEnv failed: " + e.Message); }
            }
        }

        public override IEnumerator BeginScene(Dictionary<string, object> scene)
        {
            ApplyEnvironment(Json.Obj(scene, "environment") ?? new Dictionary<string, object>());
            object player = LocalPlayer();
            if (player != null && Json.Bool(Options, "ghost_player", true))
                Reflect.Call(player, "SetGhostMode", true);  // enemies ignore the player (VERIFY)

            if (Json.Bool(Options, "spawn_enemies", false))
            {
                object zs = Reflect.StaticMember(Reflect.FindType("ZNetScene"), "instance");
                var actors = Json.Arr(scene, "actors") ?? new List<object>();
                var area = Json.Obj(scene, "area");
                float radius = (float)Math.Min(Json.Num(area, "radius", 8.0), 12.0);
                var rnd = new System.Random((int)(Json.Num(scene, "seed") % int.MaxValue));
                Vector3 center = Anchor() + new Vector3(8f, 0f, 0f);
                foreach (object a in actors)
                {
                    string prefabName = Json.Str(a as Dictionary<string, object>, "prefab", "Skeleton");
                    var prefab = Reflect.Call(zs, "GetPrefab", prefabName) as GameObject;
                    if (prefab == null) { notes.Add("prefab '" + prefabName + "' not found"); continue; }
                    double th = rnd.NextDouble() * Math.PI * 2, r = radius * Math.Sqrt(rnd.NextDouble());
                    Vector3 pos = center + new Vector3((float)(r * Math.Cos(th)), 50f, (float)(r * Math.Sin(th)));
                    RaycastHit hit;
                    if (Physics.Raycast(pos, Vector3.down, out hit, 200f)) pos = hit.point + Vector3.up * 0.1f;
                    GameObject go = UnityEngine.Object.Instantiate(prefab, pos, Quaternion.Euler(0f, (float)(rnd.NextDouble() * 360), 0f));
                    if (Json.Bool(Options, "ai_passive", true))
                        foreach (MonoBehaviour mb in go.GetComponents<MonoBehaviour>())
                            if (mb != null && (mb.GetType().Name == "MonsterAI" || mb.GetType().Name == "BaseAI")) mb.enabled = false;
                    spawned.Add(go);
                    yield return null;
                }
            }
            yield return null;
        }

        public override void EndScene()
        {
            foreach (GameObject go in spawned) if (go != null) UnityEngine.Object.Destroy(go);
            spawned.Clear();
        }

        public override List<SelfTestResult> SelfTests()
        {
            var r = new List<SelfTestResult>();
            object envMan = Reflect.StaticMember(Reflect.FindType("EnvMan"), "instance");
            r.Add(new SelfTestResult
            {
                Name = "valheim_envman", Ok = envMan != null,
                Detail = envMan != null ? "EnvMan.instance found (time of day / weather control)" : "EnvMan.instance not found",
                Hint = envMan != null ? "" : "load into a world first; if still missing, the member was renamed: environment randomization is skipped",
            });
            object player = LocalPlayer();
            r.Add(new SelfTestResult
            {
                Name = "valheim_player", Ok = player != null,
                Detail = player != null ? "local player found (used as the scene anchor)" : "no local player",
                Hint = player != null ? "" : "load into a world before running the doctor",
            });
            foreach (string n in notes) r.Add(new SelfTestResult { Name = "valheim_note", Ok = false, Detail = n });
            return r;
        }
    }
}
