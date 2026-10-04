// Finding humanoids in a running Unity game and mapping their bones onto the unified keypoint schema.
// Game-agnostic: uses only Animator / Transform / Renderer. Port of the (tested) Lua runtime's resolver.
using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace DataOpen
{
    public sealed class Actor
    {
        public int Id;
        public GameObject Root;
        public Animator Anim;
        public string RigId;
        public Dictionary<string, Transform> Bones = new Dictionary<string, Transform>();  // normalized name -> bone
        public List<string> RawNames = new List<string>();
    }

    public sealed class ResolvedSkeleton
    {
        public Vector3[] Positions;
        public bool[] Valid;
        public List<string> Unmapped = new List<string>();
    }

    public sealed class BoneResolver
    {
        public readonly List<string> Keypoints;
        // profile overrides: keypoint -> [(bone, weight), ...]
        readonly Dictionary<string, List<KeyValuePair<string, double>>> overrides =
            new Dictionary<string, List<KeyValuePair<string, double>>>();

        static readonly Dictionary<string, HumanBodyBones> Humanoid = new Dictionary<string, HumanBodyBones>
        {
            { "head", HumanBodyBones.Head }, { "neck", HumanBodyBones.Neck }, { "pelvis", HumanBodyBones.Hips },
            { "l_shoulder", HumanBodyBones.LeftUpperArm }, { "r_shoulder", HumanBodyBones.RightUpperArm },
            { "l_elbow", HumanBodyBones.LeftLowerArm }, { "r_elbow", HumanBodyBones.RightLowerArm },
            { "l_wrist", HumanBodyBones.LeftHand }, { "r_wrist", HumanBodyBones.RightHand },
            { "l_knee", HumanBodyBones.LeftLowerLeg }, { "r_knee", HumanBodyBones.RightLowerLeg },
            { "l_ankle", HumanBodyBones.LeftFoot }, { "r_ankle", HumanBodyBones.RightFoot },
        };

        // Name patterns for non-humanoid rigs (normalized: lowercase, letters/digits only, prefix up to ':' '|' '.' removed).
        // '@' stands for the side token: left/l or right/r. Mixamo, Valheim-like, Rust-like and UE-like names.
        static readonly Dictionary<string, string[]> Patterns = new Dictionary<string, string[]>
        {
            { "head", new[] { "head" } },
            { "neck", new[] { "neck", "neck1", "neck01" } },
            { "pelvis", new[] { "pelvis", "hips", "hip" } },
            { "shoulder", new[] { "@arm", "@upperarm", "upperarm@", "@shoulderjoint" } },
            { "elbow", new[] { "@forearm", "forearm@", "@lowerarm", "lowerarm@" } },
            { "wrist", new[] { "@hand", "hand@" } },
            { "knee", new[] { "@leg", "@lowerleg", "lowerleg@", "@calf", "calf@", "@knee" } },
            { "ankle", new[] { "@foot", "foot@" } },
        };

        public BoneResolver(List<string> keypoints, Dictionary<string, object> overridesJson)
        {
            Keypoints = keypoints;
            if (overridesJson == null) return;
            foreach (var kv in overridesJson)
            {
                var parts = new List<KeyValuePair<string, double>>();
                var arr = kv.Value as List<object>;
                if (arr == null) continue;
                foreach (object o in arr)
                {
                    var pair = o as List<object>;
                    if (pair == null || pair.Count == 0) continue;
                    double w = pair.Count > 1 && pair[1] is double ? (double)pair[1] : 1.0;
                    parts.Add(new KeyValuePair<string, double>(Norm(Convert.ToString(pair[0])), w));
                }
                overrides[kv.Key] = parts;
            }
        }

        public static string Norm(string name)
        {
            if (name == null) return "";
            int cut = Math.Max(name.LastIndexOf(':'), Math.Max(name.LastIndexOf('|'), name.LastIndexOf('.')));
            if (cut >= 0) name = name.Substring(cut + 1);
            var sb = new StringBuilder();
            foreach (char c in name)
                if (char.IsLetterOrDigit(c)) sb.Append(char.ToLowerInvariant(c));
            return sb.ToString();
        }

        static IEnumerable<string> Candidates(string keypoint)
        {
            string side = null;
            string part = keypoint;
            if (keypoint.StartsWith("l_")) { side = "l"; part = keypoint.Substring(2); }
            else if (keypoint.StartsWith("r_")) { side = "r"; part = keypoint.Substring(2); }
            string[] pats;
            if (!Patterns.TryGetValue(part, out pats)) yield break;
            foreach (string p in pats)
            {
                if (side == null)
                {
                    if (p.IndexOf('@') < 0) yield return p;
                    continue;
                }
                if (p.IndexOf('@') < 0) continue;
                yield return p.Replace("@", side == "l" ? "left" : "right");
                yield return p.Replace("@", side);
            }
        }

        public Actor Build(Animator anim)
        {
            var a = new Actor { Id = anim.GetInstanceID(), Anim = anim, Root = anim.gameObject };
            foreach (Transform t in anim.transform.GetComponentsInChildren<Transform>(true))
            {
                string n = Norm(t.name);
                a.RawNames.Add(t.name);
                if (n.Length > 0 && !a.Bones.ContainsKey(n)) a.Bones[n] = t;
            }
            a.RigId = anim.isHuman ? "unity_humanoid" : "unity_named";
            return a;
        }

        Transform Find(Actor a, string keypoint)
        {
            if (a.Anim != null && a.Anim.isHuman)
            {
                HumanBodyBones hb;
                if (Humanoid.TryGetValue(keypoint, out hb))
                {
                    Transform t = a.Anim.GetBoneTransform(hb);
                    if (t != null) return t;
                }
            }
            foreach (string c in Candidates(keypoint))
            {
                Transform t;
                if (a.Bones.TryGetValue(c, out t)) return t;
            }
            return null;
        }

        public ResolvedSkeleton Resolve(Actor a)
        {
            var r = new ResolvedSkeleton { Positions = new Vector3[Keypoints.Count], Valid = new bool[Keypoints.Count] };
            for (int i = 0; i < Keypoints.Count; i++)
            {
                string kp = Keypoints[i];
                List<KeyValuePair<string, double>> parts;
                if (overrides.TryGetValue(kp, out parts))
                {
                    Vector3 sum = Vector3.zero;
                    double wsum = 0;
                    bool ok = parts.Count > 0;
                    foreach (var part in parts)
                    {
                        Transform t;
                        if (!a.Bones.TryGetValue(part.Key, out t)) { ok = false; break; }
                        sum += t.position * (float)part.Value;
                        wsum += part.Value;
                    }
                    if (ok && wsum > 0) { r.Positions[i] = sum / (float)wsum; r.Valid[i] = true; continue; }
                }
                else
                {
                    Transform t = Find(a, kp);
                    if (t != null) { r.Positions[i] = t.position; r.Valid[i] = true; continue; }
                }
                r.Unmapped.Add(kp);
            }
            return r;
        }
    }

    public static class Humanoids
    {
        static readonly Dictionary<int, Actor> cache = new Dictionary<int, Actor>();

        public static void ClearCache() { cache.Clear(); }

        /// <summary>All animated humanoids within `radius` of `anchor` that resolve at least 75% of the keypoints.</summary>
        public static List<Actor> Discover(Vector3 anchor, float radius, BoneResolver resolver)
        {
            var result = new List<Actor>();
            float r2 = radius * radius;
            foreach (Animator anim in UnityEngine.Object.FindObjectsOfType<Animator>())
            {
                if (anim == null || !anim.gameObject.activeInHierarchy) continue;
                if ((anim.transform.position - anchor).sqrMagnitude > r2) continue;
                Actor a;
                if (!cache.TryGetValue(anim.GetInstanceID(), out a) || a.Root == null)
                {
                    a = resolver.Build(anim);
                    cache[a.Id] = a;
                }
                var sk = resolver.Resolve(a);
                int ok = 0;
                foreach (bool v in sk.Valid) if (v) ok++;
                if (ok >= resolver.Keypoints.Count * 0.75) result.Add(a);
            }
            result.Sort((x, y) =>
                (x.Root.transform.position - anchor).sqrMagnitude.CompareTo((y.Root.transform.position - anchor).sqrMagnitude));
            return result;
        }

        public static List<Vector3> HullPoints(Actor a)
        {
            var pts = new List<Vector3>();
            var rs = a.Root.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return pts;
            Bounds b = rs[0].bounds;
            for (int i = 1; i < rs.Length; i++) b.Encapsulate(rs[i].bounds);
            Vector3 c = b.center, e = b.extents;
            for (int sx = -1; sx <= 1; sx += 2)
                for (int sy = -1; sy <= 1; sy += 2)
                    for (int sz = -1; sz <= 1; sz += 2)
                        pts.Add(c + new Vector3(sx * e.x, sy * e.y, sz * e.z));
            return pts;
        }
    }
}
