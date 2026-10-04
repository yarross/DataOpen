// BepInEx plugin for Valheim. Mailbox: BepInEx/config/dataopen (matches profiles/valheim.toml).
using System.IO;
using BepInEx;
using UnityEngine;

namespace DataOpen.Valheim
{
    [BepInPlugin("com.dataopen.valheim", "DataOpen", "0.1.0")]
    public sealed class Plugin : BaseUnityPlugin
    {
        void Awake()
        {
            string defaultDir = Path.Combine(Paths.ConfigPath, "dataopen");
            var dir = Config.Bind("General", "MailboxDir", defaultDir, "Folder shared with the DataOpen core");
            RpcServer.Start(gameObject, dir.Value, new ValheimBindings(), msg => Logger.LogInfo(msg));
        }
    }
}
