// File mailbox, game side (docs/PROTOCOL.md): reads req.json, writes res.json.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;

namespace DataOpen
{
    public sealed class Mailbox
    {
        public readonly string Dir;
        readonly string reqPath;
        readonly string resPath;
        object lastId;

        public Mailbox(string dir)
        {
            Dir = dir;
            Directory.CreateDirectory(dir);
            reqPath = Path.Combine(dir, "req.json");
            resPath = Path.Combine(dir, "res.json");
        }

        /// <summary>Returns a new request, or null (nothing new, or the core is mid-write).</summary>
        public Dictionary<string, object> TryReadRequest()
        {
            try
            {
                if (!File.Exists(reqPath)) return null;
                string text;
                using (var fs = new FileStream(reqPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var sr = new StreamReader(fs, Encoding.UTF8))
                    text = sr.ReadToEnd();
                if (text.Length == 0) return null;
                var req = Json.Parse(text) as Dictionary<string, object>;
                if (req == null || !req.ContainsKey("id")) return null;
                if (Equals(req["id"], lastId)) return null;
                lastId = req["id"];
                return req;
            }
            catch (IOException) { return null; }
            catch (FormatException) { return null; }
            catch (UnauthorizedAccessException) { return null; }
        }

        public void WriteResult(object id, Dictionary<string, object> result)
        {
            Write(new Dictionary<string, object> { { "id", id }, { "result", result ?? new Dictionary<string, object>() } });
        }

        public void WriteError(object id, string type, string message)
        {
            var err = new Dictionary<string, object> { { "message", message }, { "type", type } };
            Write(new Dictionary<string, object> { { "id", id }, { "error", err } });
        }

        void Write(Dictionary<string, object> message)
        {
            string text;
            try { text = Json.Serialize(message); }
            catch (Exception e)
            {
                var err = new Dictionary<string, object>
                {
                    { "message", "cannot encode the response: " + e.Message }, { "type", "EncodeError" }
                };
                text = Json.Serialize(new Dictionary<string, object> { { "id", message["id"] }, { "error", err } });
            }
            string tmp = resPath + ".tmp";
            File.WriteAllText(tmp, text, new UTF8Encoding(false));
            // Windows: replacing the file fails while the core has it open for reading. Retry for ~1.5 s.
            for (int attempt = 0; ; attempt++)
            {
                try
                {
                    if (File.Exists(resPath)) File.Delete(resPath);
                    File.Move(tmp, resPath);
                    return;
                }
                catch (Exception e)
                {
                    if (!(e is IOException) && !(e is UnauthorizedAccessException)) throw;
                    if (attempt >= 300) throw;
                    Thread.Sleep(5);
                }
            }
        }
    }
}
