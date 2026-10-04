// MonoBehaviour that polls the mailbox every frame and runs one request at a time as a coroutine.
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DataOpen
{
    public sealed class RpcServer : MonoBehaviour
    {
        Mailbox box;
        UnityHost host;
        Action<string> log;
        Coroutine current;
        Call call;
        float startedAt;
        bool busy;
        public float TimeoutSeconds = 120f;

        public static RpcServer Start(GameObject owner, string mailboxDir, IGameBindings bindings, Action<string> log)
        {
            var s = owner.AddComponent<RpcServer>();
            s.log = log ?? delegate { };
            s.box = new Mailbox(mailboxDir);
            s.host = new UnityHost(bindings, mailboxDir, s.log);
            s.log("DataOpen RPC ready, mailbox " + mailboxDir);
            return s;
        }

        void Update()
        {
            if (box == null) return;
            if (busy)
            {
                if (Time.realtimeSinceStartup - startedAt > TimeoutSeconds) Abort("Timeout", "handler timed out after " + TimeoutSeconds + "s");
                return;
            }
            Dictionary<string, object> req = box.TryReadRequest();
            if (req == null) return;
            busy = true;
            startedAt = Time.realtimeSinceStartup;
            call = new Call { Req = req, Params = Json.Obj(req, "params") ?? new Dictionary<string, object>() };
            current = StartCoroutine(Run(call));
        }

        IEnumerator Run(Call c)
        {
            if ((int)Json.Num(c.Req, "v", -1) != UnityHost.Protocol)
            {
                c.Fail("ProtocolError", "protocol version " + Json.Num(c.Req, "v", -1) + " != " + UnityHost.Protocol);
            }
            else
            {
                IEnumerator handler = host.Handle(c);
                while (true)
                {
                    bool more;
                    try { more = handler.MoveNext(); }
                    catch (Exception e) { c.Fail(e.GetType().Name, e.ToString()); break; }
                    if (!more) break;
                    yield return handler.Current;
                }
            }
            Respond(c);
        }

        void Respond(Call c)
        {
            object id = c.Req["id"];
            try
            {
                if (c.Failed) box.WriteError(id, c.ErrorType, c.ErrorMessage);
                else box.WriteResult(id, c.Result);
            }
            catch (Exception e) { log("could not write the response: " + e.Message); }
            busy = false;
            current = null;
        }

        void Abort(string type, string message)
        {
            if (current != null) StopCoroutine(current);
            call.Fail(type, message);
            log(message);
            Respond(call);
        }
    }
}
