// DataOpen.Contracts — C# mirror of dataopen/core/models.py. NOT COMPILED/TESTED in this repo yet
// (no .NET SDK in the authoring environment); treat as a reviewed sketch, not a verified build.
//
// Conventions (adapter converts engine-native values BEFORE sending):
//   units = meters; camera = OpenCV (x right, y down, z forward), extrinsics world->camera;
//   pixel coords are continuous, origin top-left corner of the top-left pixel.
using System.Collections.Generic;

namespace DataOpen.Contracts
{
    public enum Visibility : sbyte { OutOfFrame = 0, Occluded = 1, Visible = 2 }
    public enum FrameKind { Positive, Negative }

    public sealed class CameraModel
    {
        public int Width, Height;
        public float Fx, Fy, Cx, Cy;
        public float[] WorldToCamera = new float[16];   // row-major 4x4, OpenCV convention
        public float Near = 0.1f;
    }

    public sealed class EntityHandle
    {
        public int EntityId;
        public string RigId;
        public Dictionary<string, object> Meta = new Dictionary<string, object>(); // skin, outfit, weapon...
    }

    public sealed class EntityState
    {
        public int EntityId;
        public string RigId;
        public float[] SkeletonWorld;      // K*3, schema order (core sends the schema at handshake)
        public bool[] JointValid;          // K
        public float[] HullPointsWorld;    // optional, M*3 (mesh bounds -> tight bbox)
        public sbyte[] EngineVisibility;   // optional, K; 1/2 from engine raycasts
    }

    public sealed class FrameSnapshot
    {
        public string FrameToken;          // same token the core used in CaptureRequest
        public long Tick;
        public CameraModel Camera;
        public List<EntityState> Entities = new List<EntityState>();
        public string DepthPath;           // optional: raw float32 z-depth of occluders, shared memory/file
        public byte[] Thumbnail;           // optional 9x8 grayscale for stale-frame detection
    }

    // The sampled values arrive as plain dictionaries; the adapter only APPLIES them.
    public sealed class SceneSpec
    {
        public int SceneIndex; public long Seed; public string Split;
        public Dictionary<string, object> Environment;
        public List<Dictionary<string, object>> Actors;
        public float[] AreaCenter; public float AreaRadius, MinSeparation;
    }

    public sealed class CameraSpec
    {
        public float Distance, YawDeg, PitchDeg, RollDeg, FovDeg, HeightOffset;
        public int TargetIndex;
    }

    public sealed class FrameSpec
    {
        public int SceneIndex, FrameIndex; public long Seed; public FrameKind Kind;
        public CameraSpec Camera;
        public List<Dictionary<string, object>> ActorFrame;
    }

    public sealed class CaptureRequest
    {
        public string FrameId; public FrameSpec Frame; public int Width, Height; public bool WantDepth;
    }
}
