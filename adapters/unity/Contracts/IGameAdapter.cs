// DataOpen.Contracts — C# mirror of dataopen/core/interfaces.py (see Models.cs for status).
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace DataOpen.Contracts
{
    public class AdapterException : Exception { public AdapterException(string m) : base(m) { } }

    public interface IEntitySpawner
    {
        IReadOnlyList<EntityHandle> Spawn(SceneSpec scene);
        void UpdateActors(IReadOnlyList<EntityHandle> handles, FrameSpec frame);
        void SetActive(IReadOnlyList<EntityHandle> handles, bool active);   // negative frames
        void DespawnAll();   // prefer pooling: deactivate, don't Destroy
    }

    public interface IEnvironmentController { void Apply(SceneSpec scene); }

    public interface ISkeletonExtractor
    {
        IReadOnlyList<string> Schema { get; }                  // unified keypoint names, in order
        BoneMapping MappingFor(string rigId);
        // MUST run in the same engine instant as the pixel capture (end of frame, simulation paused).
        IReadOnlyList<EntityState> Extract(IReadOnlyList<EntityHandle> handles);
    }

    public interface ICaptureBridge
    {
        // Place camera, settle/step simulation N fixed ticks, render, read bones. Pixels stay engine-side.
        Task<FrameSnapshot> CaptureAsync(CaptureRequest request);
        Task CommitAsync(FrameSnapshot snapshot, string destPath);   // encode + write ACCEPTED frame
        void Discard(FrameSnapshot snapshot);                        // free REJECTED frame
    }

    public interface IGameAdapter
    {
        string Name { get; }
        IEntitySpawner Spawner { get; }
        IEnvironmentController Environment { get; }
        ISkeletonExtractor Extractor { get; }
        ICaptureBridge Capture { get; }
    }

    // Unified keypoint -> weighted internal bones. pelvis = 0.5*Thigh_L + 0.5*Thigh_R needs no special code.
    public sealed class BoneMapping
    {
        public string RigId;
        public Dictionary<string, (string bone, float weight)[]> Weights;

        public bool TryResolve(string keypoint, Func<string, UnityEngine.Vector3?> boneWorldPos,
                               out UnityEngine.Vector3 result)
        {
            result = default;
            if (!Weights.TryGetValue(keypoint, out var parts)) return false;
            UnityEngine.Vector3 sum = default; float wsum = 0;
            foreach (var (bone, w) in parts)
            {
                var p = boneWorldPos(bone);
                if (p == null) return false;        // any missing bone invalidates the joint
                sum += p.Value * w; wsum += w;
            }
            result = sum / wsum;
            return true;
        }
    }
}
