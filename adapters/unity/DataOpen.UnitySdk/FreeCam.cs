// A free camera that renders the live game scene into a RenderTexture: independent of the game's own
// camera, HUD and window size. Unity's Camera.fieldOfView is the VERTICAL FOV, which is what the protocol wants.
using System;
using System.Collections.Generic;
using UnityEngine;

namespace DataOpen
{
    public sealed class FreeCam
    {
        GameObject go;
        Camera cam;
        RenderTexture rt;
        int width, height;
        static readonly RaycastHit[] hits = new RaycastHit[32];

        public Camera Cam { get { return cam; } }

        public void Ensure(int w, int h)
        {
            if (cam == null)
            {
                go = new GameObject("DataOpenCamera");
                UnityEngine.Object.DontDestroyOnLoad(go);
                cam = go.AddComponent<Camera>();
                Camera main = Camera.main;
                if (main != null) cam.CopyFrom(main);  // sky, culling mask, clip planes, HDR... same look as the game
                cam.enabled = false;                    // we render manually
                cam.depth = -100;
            }
            if (rt == null || w != width || h != height)
            {
                if (rt != null) { rt.Release(); UnityEngine.Object.Destroy(rt); }
                rt = new RenderTexture(w, h, 24, RenderTextureFormat.ARGB32);
                width = w;
                height = h;
            }
            cam.targetTexture = rt;
            cam.aspect = (float)w / h;
        }

        public void Place(Vector3 pos, Vector3 lookAt, float rollDeg, float verticalFov)
        {
            Vector3 fwd = (lookAt - pos).normalized;
            Quaternion rot = Quaternion.LookRotation(fwd, Vector3.up);
            rot = Quaternion.AngleAxis(rollDeg, fwd) * rot;
            cam.transform.SetPositionAndRotation(pos, rot);
            cam.fieldOfView = verticalFov;
        }

        /// <summary>Engine-native projection: top-left origin pixels; false when behind the camera or off screen.</summary>
        public bool Project(Vector3 world, out float u, out float v)
        {
            Vector3 sp = cam.WorldToScreenPoint(world);  // bottom-left origin
            u = sp.x;
            v = height - sp.y;
            return sp.z > 0.01f && u >= 0 && u < width && v >= 0 && v < height;
        }

        /// <summary>Render now (call after WaitForEndOfFrame, in the same frame the bones were read).</summary>
        public Texture2D Render()
        {
            RenderTexture prev = RenderTexture.active;
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(width, height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            tex.Apply(false);
            RenderTexture.active = prev;
            return tex;
        }

        public void Dispose()
        {
            if (rt != null) { rt.Release(); UnityEngine.Object.Destroy(rt); rt = null; }
            if (go != null) { UnityEngine.Object.Destroy(go); go = null; cam = null; }
        }

        /// <summary>Distance to the first solid, non-trigger collider on the segment that does not belong to
        /// `ignore` (the actor's own hierarchy); a negative number when the segment is clear.</summary>
        public static float FirstBlocker(Vector3 from, Vector3 to, Transform ignore)
        {
            Vector3 d = to - from;
            float dist = d.magnitude;
            if (dist < 1e-4f) return -1f;
            int n = Physics.RaycastNonAlloc(from, d / dist, hits, dist, ~0, QueryTriggerInteraction.Ignore);
            float best = float.MaxValue;
            for (int i = 0; i < n; i++)
            {
                if (ignore != null && hits[i].collider.transform.IsChildOf(ignore)) continue;
                if (hits[i].distance < best) best = hits[i].distance;
            }
            return best == float.MaxValue ? -1f : best;
        }
    }
}
