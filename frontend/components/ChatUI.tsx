"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import { Send, Bot, User, ExternalLink, RefreshCw } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface Source { url: string; title: string; section: string; }
interface Message { role: "user" | "assistant"; content: string; sources?: Source[]; loading?: boolean; }

const MAROON = "#500000";
const BG = "#ffffff";
const CARD = "#f5f5f5";
const BORDER = "#e5e5e5";

const SECTION_OPTIONS = [
  { value: "", label: "All" },
  { value: "people", label: "People" },
  { value: "research", label: "Research" },
  { value: "academics", label: "Academics" },
  { value: "admissions", label: "Admissions" },
  { value: "news", label: "News" },
  { value: "about", label: "About" },
];

const SECTION_COLORS: Record<string, { bg: string; color: string }> = {
  people:     { bg: "#f3e8ff", color: "#7e22ce" },
  research:   { bg: "#dbeafe", color: "#1d4ed8" },
  academics:  { bg: "#dcfce7", color: "#15803d" },
  admissions: { bg: "#fef9c3", color: "#a16207" },
  news:       { bg: "#fee2e2", color: "#b91c1c" },
  events:     { bg: "#ffedd5", color: "#c2410c" },
  about:      { bg: "#f3f4f6", color: "#4b5563" },
};

const TOPICS = [
  { label: "Research areas",     prompt: "What research areas does TAMU ECE specialize in?" },
  { label: "Faculty",            prompt: "Who are the faculty members in TAMU ECE?" },
  { label: "Graduate programs",  prompt: "What graduate programs are offered in TAMU ECE?" },
  { label: "Admissions",         prompt: "How do I apply to TAMU ECE?" },
  { label: "Patents",            prompt: "What patents has TAMU ECE filed?" },
  { label: "Scholarships",       prompt: "What scholarships and financial aid are available?" },
];

export default function ChatUI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [section, setSection] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const hasMessages = messages.length > 0;

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function send(question: string) {
    if (!question.trim() || streaming) return;
    setInput("");
    setMessages(prev => [...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", loading: true },
    ]);
    setStreaming(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, section_filter: section || undefined }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "", sources: Source[] = [], answer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n"); buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") break;
          try { const p = JSON.parse(data); if (Array.isArray(p)) { sources = p; continue; } } catch {}
          answer += data;
          setMessages(prev => { const u = [...prev]; u[u.length - 1] = { role: "assistant", content: answer, sources, loading: false }; return u; });
        }
      }
    } catch {
      setMessages(prev => { const u = [...prev]; u[u.length - 1] = { role: "assistant", content: "Sorry, something went wrong. Please try again.", loading: false }; return u; });
    } finally { setStreaming(false); }
  }

  function onSubmit(e: FormEvent) { e.preventDefault(); send(input.trim()); }

  /* ── LANDING ── */
  if (!hasMessages) return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100vh", width: "100%", backgroundColor: BG, fontFamily: "system-ui, sans-serif" }}>

      <h1 style={{ color: "#111111", fontSize: "2.25rem", fontWeight: 400, marginBottom: "2.5rem", letterSpacing: "-0.02em" }}>
        What's on your mind today?
      </h1>

      <form onSubmit={onSubmit} style={{ width: "100%", maxWidth: "640px", padding: "0 1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", backgroundColor: CARD, border: `1px solid ${BORDER}`, borderRadius: "999px", padding: "14px 20px", boxShadow: "0 1px 6px rgba(0,0,0,0.08)" }}>
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask anything"
            style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#111111", fontSize: "1rem" }}
          />
          <button type="submit" disabled={!input.trim()} style={{ display: "flex", alignItems: "center", justifyContent: "center", width: "36px", height: "36px", borderRadius: "50%", backgroundColor: input.trim() ? MAROON : BORDER, border: "none", cursor: input.trim() ? "pointer" : "default", flexShrink: 0, transition: "background 0.2s" }}>
            <Send size={15} color="white" />
          </button>
        </div>
      </form>

      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "10px", marginTop: "1.5rem", maxWidth: "640px", padding: "0 1rem" }}>
        {TOPICS.map(t => (
          <button key={t.label} onClick={() => send(t.prompt)}
            style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px 16px", borderRadius: "999px", backgroundColor: CARD, border: `1px solid ${BORDER}`, color: "#111111", fontSize: "0.875rem", cursor: "pointer" }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.backgroundColor = MAROON; (e.currentTarget as HTMLButtonElement).style.color = "white"; (e.currentTarget as HTMLButtonElement).style.borderColor = MAROON; }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.backgroundColor = CARD; (e.currentTarget as HTMLButtonElement).style.color = "#111111"; (e.currentTarget as HTMLButtonElement).style.borderColor = BORDER; }}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );

  /* ── CHAT ── */
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", backgroundColor: BG, fontFamily: "system-ui, sans-serif" }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 24px", borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{ width: "30px", height: "30px", borderRadius: "50%", backgroundColor: MAROON, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Bot size={15} color="white" />
          </div>
          <span style={{ color: "#111111", fontWeight: 600, fontSize: "0.9rem" }}>TAMU ECE Assistant</span>
        </div>
        <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
          {SECTION_OPTIONS.map(o => (
            <button key={o.value} onClick={() => setSection(o.value)}
              style={{ padding: "4px 12px", borderRadius: "999px", fontSize: "0.75rem", border: `1px solid ${section === o.value ? MAROON : BORDER}`, backgroundColor: section === o.value ? MAROON : "transparent", color: section === o.value ? "white" : "#111111", cursor: "pointer" }}>
              {o.label}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px 16px" }}>
        <div style={{ maxWidth: "720px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "20px" }}>
          {messages.map((msg, i) => (
            <div key={i} style={{ display: "flex", gap: "12px", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
              {msg.role === "assistant" && (
                <div style={{ width: "30px", height: "30px", borderRadius: "50%", backgroundColor: MAROON, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: "4px" }}>
                  <Bot size={14} color="white" />
                </div>
              )}
              <div style={{ maxWidth: "75%" }}>
                <div style={{ padding: "12px 16px", borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "4px 18px 18px 18px", backgroundColor: msg.role === "user" ? MAROON : CARD, border: msg.role === "assistant" ? `1px solid ${BORDER}` : "none", color: msg.role === "user" ? "white" : "#111111", fontSize: "0.875rem", lineHeight: 1.6 }}>
                  {msg.loading ? (
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#6b7280" }}>
                      <RefreshCw size={13} style={{ animation: "spin 1s linear infinite" }} />
                      <span>Searching the department…</span>
                    </div>
                  ) : msg.role === "user" ? (
                    msg.content
                  ) : (
                    <div className="markdown-body">
                      <ReactMarkdown>{msg.content.replace(/\\n/g, "\n")}</ReactMarkdown>
                    </div>
                  )}
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div style={{ marginTop: "8px", display: "flex", flexDirection: "column", gap: "4px" }}>
                    <span style={{ fontSize: "0.7rem", color: "#6b7280", paddingLeft: "4px" }}>Sources</span>
                    {msg.sources.map((s, si) => {
                      const col = SECTION_COLORS[s.section] ?? SECTION_COLORS.about;
                      return (
                        <a key={si} href={s.url} target="_blank" rel="noopener noreferrer"
                          style={{ display: "flex", alignItems: "flex-start", gap: "8px", padding: "8px 12px", borderRadius: "8px", backgroundColor: CARD, border: `1px solid ${BORDER}`, color: "#6b7280", fontSize: "0.75rem", textDecoration: "none" }}>
                          <ExternalLink size={11} style={{ marginTop: "2px", flexShrink: 0 }} />
                          <div>
                            <span style={{ color: "#111111", fontWeight: 500 }}>{s.title}</span>
                            <span style={{ marginLeft: "8px", padding: "1px 6px", borderRadius: "4px", fontSize: "0.65rem", backgroundColor: col.bg, color: col.color }}>{s.section}</span>
                          </div>
                        </a>
                      );
                    })}
                  </div>
                )}
              </div>
              {msg.role === "user" && (
                <div style={{ width: "30px", height: "30px", borderRadius: "50%", backgroundColor: BORDER, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: "4px" }}>
                  <User size={14} color="#9ca3af" />
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div style={{ padding: "16px", borderTop: `1px solid ${BORDER}`, flexShrink: 0 }}>
        <form onSubmit={onSubmit} style={{ maxWidth: "720px", margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", backgroundColor: CARD, border: `1px solid ${BORDER}`, borderRadius: "999px", padding: "12px 18px" }}>
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Ask anything"
              disabled={streaming}
              style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#111111", fontSize: "0.9rem" }}
            />
            <button type="submit" disabled={streaming || !input.trim()}
              style={{ display: "flex", alignItems: "center", justifyContent: "center", width: "34px", height: "34px", borderRadius: "50%", backgroundColor: !streaming && input.trim() ? MAROON : "#2a2a2a", border: "none", cursor: !streaming && input.trim() ? "pointer" : "default", flexShrink: 0, transition: "background 0.2s" }}>
              <Send size={14} color="white" />
            </button>
          </div>
        </form>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
