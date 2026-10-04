// Reference sketch of the Unity-specific pieces. NOT COMPILED/TESTED (see Models.cs).
// Only the parts that are easy to get wrong are written out.
using System.Collections;
using UnityEngine;

namespace DataOpen.UnityAdapter
{
    public static class UnityCameraExport
    {
        // Unity: left-handed world, camera looks down +Z in worldToCameraMatrix's *OpenGL* view space
        // (right-handed, y up, looking down -Z). OpenCV camera = x right, y down, z forward, so the
        // conversion is a flip of the y and z axes of view space. Reflections are fine for projection.
        public static float[] WorldToCameraCv(Camera cam)
        {
            Matrix4x4 gl = cam.worldToCameraMatrix;
            Matrix4x4 flip = Matrix4x4.Scale(new Vector3(1, -1, -1));
            Matrix4x4 cv = flip * gl;
            var m = new float[16];
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) m[r * 4 + c] = cv[r, c];
            return m;
        }

        // fx, fy from the projection matrix; lens shift / physical camera offsets are IGNORED here
        // (cx, cy = image centre). If you use lens shift, derive cx/cy from P[0,2], P[1,2] and test it.
        public static (float fx, float fy, float cx, float cy) Intrinsics(Camera cam, int w, int h)
        {
            Matrix4x4 p = cam.projectionMatrix;
            return (p[0, 0] * w / 2f, p[1, 1] * h / 2f, w / 2f, h / 2f);
        }
    }

    public static class UnityHumanoidExtractor
    {
        // For Humanoid-avatar rigs Unity already normalises bone semantics across models:
        // no per-rig name mapping needed. Generic rigs fall back to BoneMapping.
        public static Vector3? Bone(Animator a, HumanBodyBones b)
        {
            var t = a.GetBoneTransform(b);
            return t ? t.position : (Vector3?)null;
        }
    }

    public class UnityCaptureLoop : MonoBehaviour
    {
        // Pattern: pause simulation, step a fixed number of ticks so animation/physics/LODs/TAA settle,
        // wait for end of frame, then read bones and issue the GPU readback in the SAME frame.
        public IEnumerator CaptureOne(System.Action<FrameResult> done, int settleTicks = 3)
        {
            Time.timeScale = 1f;
            Time.fixedDeltaTime = 1f / 60f;
            for (int i = 0; i < settleTicks; i++) yield return new WaitForFixedUpdate();
            yield return new WaitForEndOfFrame();
            Time.timeScale = 0f;
            // 1) read bone transforms now (cheap)  2) AsyncGPUReadback.Request(renderTexture, ...)
            //    keyed by frame token so the readback may land 1-3 frames later without losing sync.
            // 3) Do NOT read bones on the server/host: replication lags the rendered client frame.
            done(new FrameResult());
        }
        public class FrameResult { }
    }
}
