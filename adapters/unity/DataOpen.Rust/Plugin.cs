// BepInEx plugin for the Rust CLIENT (own local server, EAC disabled). Mailbox: BepInEx/config/dataopen.
// Time of day and weather are controlled on the SERVER over RCON by the Python core (see adapters/rust_rcon.py);
// this plugin only observes the humanoids the client sees (players, scientists) and renders them with a free camera.
using System.IO;
using BepInEx;
using UnityEngine;

namespace DataOpen.Rust
{
    [BepInPlugin("com.dataopen.rust", "DataOpen", "0.1.0")]
    public sealed class Plugin : BaseUnityPlugin
    {
        void Awake()
        {
            string defaultDir = Path.Combine(Paths.ConfigPath, "dataopen");
            var dir = Config.Bind("General", "MailboxDir", defaultDir, "Folder shared with the DataOpen core");
            RpcServer.Start(gameObject, dir.Value, new RustBindings(), msg => Logger.LogInfo(msg));
        }
    }

    public sealed class RustBindings : GenericBindings
    {
        public override string GameName { get { return "Rust"; } }
    }
}
