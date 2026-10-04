// Everything game-specific lives behind this interface. The SDK works without any of it ("observe" mode:
// it finds the humanoids that exist in the scene); bindings add control over time/weather/population.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

namespace DataOpen
{
    public sealed class SelfTestResult
    {
        public string Name;
        public bool Ok;
        public string Detail = "";
        public string Hint = "";
    }

    public interface IGameBindings
    {
        string GameName { get; }
        string GameVersion();
        /// <summary>Protocol-format parameter space: { "environment": {...}, "actor": {...}, "actor_frame": {...} }.</summary>
        Dictionary<string, object> ParameterSpace();
        void Init(Dictionary<string, object> options, Action<string> log);
        /// <summary>Start of a scene: apply the environment and (optionally) spawn actors. May take several frames.</summary>
        IEnumerator BeginScene(Dictionary<string, object> scene);
        void EndScene();
        /// <summary>Where to look for humanoids and where to aim the camera when none is found.</summary>
        Vector3 Anchor();
        List<SelfTestResult> SelfTests();
    }

    /// <summary>No game knowledge at all: observe whatever humanoids are around the main camera.</summary>
    public class GenericBindings : IGameBindings
    {
        protected Action<string> Log = delegate { };
        protected Dictionary<string, object> Options = new Dictionary<string, object>();

        public virtual string GameName { get { return Application.productName; } }
        public virtual string GameVersion() { return Application.version; }
        public virtual Dictionary<string, object> ParameterSpace()
        {
            return new Dictionary<string, object>
            {
                { "environment", new Dictionary<string, object>() },
                { "actor", new Dictionary<string, object>() },
                { "actor_frame", new Dictionary<string, object>() },
            };
        }
        public virtual void Init(Dictionary<string, object> options, Action<string> log)
        {
            Options = options ?? new Dictionary<string, object>();
            Log = log ?? Log;
        }
        public virtual IEnumerator BeginScene(Dictionary<string, object> scene) { yield break; }
        public virtual void EndScene() { }
        public virtual Vector3 Anchor()
        {
            Camera main = Camera.main;
            return main != null ? main.transform.position : Vector3.zero;
        }
        public virtual List<SelfTestResult> SelfTests() { return new List<SelfTestResult>(); }
    }

    /// <summary>Reflection helpers: game-specific members are touched by NAME at runtime, so the plugin builds
    /// without referencing any game assembly and a wrong/renamed member shows up in `doctor`, not as a build error.</summary>
    public static class Reflect
    {
        const BindingFlags All = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;

        public static Type FindType(string name)
        {
            foreach (Assembly asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type t = asm.GetType(name, false);
                if (t != null) return t;
            }
            return null;
        }

        public static object StaticMember(Type t, string name)
        {
            if (t == null) return null;
            FieldInfo f = t.GetField(name, All);
            if (f != null) return f.GetValue(null);
            PropertyInfo p = t.GetProperty(name, All);
            return p != null ? p.GetValue(null, null) : null;
        }

        public static bool SetMember(object target, string name, object value)
        {
            if (target == null) return false;
            Type t = target.GetType();
            FieldInfo f = t.GetField(name, All);
            if (f != null) { f.SetValue(target, Convert.ChangeType(value, f.FieldType)); return true; }
            PropertyInfo p = t.GetProperty(name, All);
            if (p != null && p.CanWrite) { p.SetValue(target, Convert.ChangeType(value, p.PropertyType), null); return true; }
            return false;
        }

        public static object GetMember(object target, string name)
        {
            if (target == null) return null;
            Type t = target.GetType();
            FieldInfo f = t.GetField(name, All);
            if (f != null) return f.GetValue(target);
            PropertyInfo p = t.GetProperty(name, All);
            return p != null ? p.GetValue(target, null) : null;
        }

        public static object Call(object target, string method, params object[] args)
        {
            if (target == null) return null;
            MethodInfo m = target.GetType().GetMethod(method, All);
            return m != null ? m.Invoke(target, args) : null;
        }

        public static object CallStatic(Type t, string method, params object[] args)
        {
            if (t == null) return null;
            MethodInfo m = t.GetMethod(method, All);
            return m != null ? m.Invoke(null, args) : null;
        }
    }
}
