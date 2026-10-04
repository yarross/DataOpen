// Minimal JSON for the DataOpen protocol (no external dependencies; C# 7.3 / .NET Framework 4.x / Unity Mono).
// Values map to: null, bool, double, string, List<object> (arrays), Dictionary<string, object> (objects).
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace DataOpen
{
    public static class Json
    {
        public static object Parse(string text)
        {
            var p = new Parser(text);
            object v = p.ReadValue();
            p.SkipWhitespace();
            if (p.Pos < text.Length) throw new FormatException("trailing data at " + p.Pos);
            return v;
        }

        public static string Serialize(object value)
        {
            var sb = new StringBuilder();
            Write(sb, value);
            return sb.ToString();
        }

        // ---- typed accessors (protocol messages are loosely typed) ----
        public static Dictionary<string, object> Obj(object o, string key)
        {
            var d = o as Dictionary<string, object>;
            object v;
            if (d != null && d.TryGetValue(key, out v)) return v as Dictionary<string, object>;
            return null;
        }

        public static List<object> Arr(object o, string key)
        {
            var d = o as Dictionary<string, object>;
            object v;
            if (d != null && d.TryGetValue(key, out v)) return v as List<object>;
            return null;
        }

        public static string Str(object o, string key, string def = null)
        {
            var d = o as Dictionary<string, object>;
            object v;
            if (d != null && d.TryGetValue(key, out v) && v is string) return (string)v;
            return def;
        }

        public static double Num(object o, string key, double def = 0)
        {
            var d = o as Dictionary<string, object>;
            object v;
            if (d != null && d.TryGetValue(key, out v) && v is double) return (double)v;
            return def;
        }

        public static bool Bool(object o, string key, bool def = false)
        {
            var d = o as Dictionary<string, object>;
            object v;
            if (d != null && d.TryGetValue(key, out v) && v is bool) return (bool)v;
            return def;
        }

        // ---- writer ----
        static void Write(StringBuilder sb, object v)
        {
            if (v == null) { sb.Append("null"); return; }
            if (v is bool) { sb.Append((bool)v ? "true" : "false"); return; }
            if (v is string) { WriteString(sb, (string)v); return; }
            if (v is double || v is float || v is int || v is long || v is short || v is byte)
            {
                double d = Convert.ToDouble(v, CultureInfo.InvariantCulture);
                if (double.IsNaN(d) || double.IsInfinity(d)) throw new FormatException("cannot encode a non-finite number");
                if (d == Math.Floor(d) && Math.Abs(d) < 1e15) sb.Append(((long)d).ToString(CultureInfo.InvariantCulture));
                else sb.Append(d.ToString("R", CultureInfo.InvariantCulture));
                return;
            }
            var dict = v as IDictionary;
            if (dict != null)
            {
                sb.Append('{');
                bool first = true;
                foreach (DictionaryEntry e in dict)
                {
                    if (e.Value == null) continue;
                    if (!first) sb.Append(',');
                    first = false;
                    WriteString(sb, Convert.ToString(e.Key, CultureInfo.InvariantCulture));
                    sb.Append(':');
                    Write(sb, e.Value);
                }
                sb.Append('}');
                return;
            }
            var list = v as IEnumerable;
            if (list != null)
            {
                sb.Append('[');
                bool first = true;
                foreach (object item in list)
                {
                    if (!first) sb.Append(',');
                    first = false;
                    Write(sb, item);
                }
                sb.Append(']');
                return;
            }
            throw new FormatException("cannot encode " + v.GetType().Name);
        }

        static void WriteString(StringBuilder sb, string s)
        {
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4"));
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }

        // ---- parser ----
        sealed class Parser
        {
            readonly string s;
            public int Pos;

            public Parser(string text) { s = text; }

            public void SkipWhitespace()
            {
                while (Pos < s.Length && (s[Pos] == ' ' || s[Pos] == '\t' || s[Pos] == '\r' || s[Pos] == '\n')) Pos++;
            }

            public object ReadValue()
            {
                SkipWhitespace();
                if (Pos >= s.Length) throw new FormatException("unexpected end");
                char c = s[Pos];
                if (c == '{') return ReadObject();
                if (c == '[') return ReadArray();
                if (c == '"') return ReadString();
                if (Match("true")) return true;
                if (Match("false")) return false;
                if (Match("null")) return null;
                return ReadNumber();
            }

            bool Match(string word)
            {
                if (string.CompareOrdinal(s, Pos, word, 0, word.Length) != 0) return false;
                Pos += word.Length;
                return true;
            }

            Dictionary<string, object> ReadObject()
            {
                var d = new Dictionary<string, object>();
                Pos++;
                SkipWhitespace();
                if (Pos < s.Length && s[Pos] == '}') { Pos++; return d; }
                while (true)
                {
                    SkipWhitespace();
                    if (Pos >= s.Length || s[Pos] != '"') throw new FormatException("expected a string key at " + Pos);
                    string key = ReadString();
                    SkipWhitespace();
                    if (Pos >= s.Length || s[Pos] != ':') throw new FormatException("expected ':' at " + Pos);
                    Pos++;
                    d[key] = ReadValue();
                    SkipWhitespace();
                    if (Pos >= s.Length) throw new FormatException("unterminated object");
                    if (s[Pos] == '}') { Pos++; return d; }
                    if (s[Pos] != ',') throw new FormatException("expected ',' or '}' at " + Pos);
                    Pos++;
                }
            }

            List<object> ReadArray()
            {
                var l = new List<object>();
                Pos++;
                SkipWhitespace();
                if (Pos < s.Length && s[Pos] == ']') { Pos++; return l; }
                while (true)
                {
                    l.Add(ReadValue());
                    SkipWhitespace();
                    if (Pos >= s.Length) throw new FormatException("unterminated array");
                    if (s[Pos] == ']') { Pos++; return l; }
                    if (s[Pos] != ',') throw new FormatException("expected ',' or ']' at " + Pos);
                    Pos++;
                }
            }

            string ReadString()
            {
                var sb = new StringBuilder();
                Pos++;
                while (true)
                {
                    if (Pos >= s.Length) throw new FormatException("unterminated string");
                    char c = s[Pos++];
                    if (c == '"') return sb.ToString();
                    if (c != '\\') { sb.Append(c); continue; }
                    if (Pos >= s.Length) throw new FormatException("bad escape");
                    char e = s[Pos++];
                    switch (e)
                    {
                        case 'b': sb.Append('\b'); break;
                        case 'f': sb.Append('\f'); break;
                        case 'n': sb.Append('\n'); break;
                        case 'r': sb.Append('\r'); break;
                        case 't': sb.Append('\t'); break;
                        case '"': sb.Append('"'); break;
                        case '\\': sb.Append('\\'); break;
                        case '/': sb.Append('/'); break;
                        case 'u':
                            if (Pos + 4 > s.Length) throw new FormatException("bad \\u escape");
                            sb.Append((char)int.Parse(s.Substring(Pos, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                            Pos += 4;
                            break;
                        default: throw new FormatException("bad escape \\" + e);
                    }
                }
            }

            double ReadNumber()
            {
                int start = Pos;
                while (Pos < s.Length && "+-0123456789.eE".IndexOf(s[Pos]) >= 0) Pos++;
                if (start == Pos) throw new FormatException("unexpected character at " + Pos);
                return double.Parse(s.Substring(start, Pos - start), NumberStyles.Float, CultureInfo.InvariantCulture);
            }
        }
    }
}
