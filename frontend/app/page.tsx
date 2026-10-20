"use client";

import { useEffect, useRef, useState } from "react";
import { CommercePanel } from "@/components/commerce-panel";
import { ChatComposer } from "@/components/chat-composer";
import { ChatMessage, type Message, type ToolExecution, type AgentEvent } from "@/components/chat-message";

const suggestions = ["查询 ORD00001 的物流，并确认这款耳机能否连接 Windows 笔记本", "找适合跑步、有库存、500 元以内的耳机", "查询 ORD00006 的退款进度，并解释退货政策"];

const SESSION_KEY = "commerce-session-id";
async function api(path: string, init?: RequestInit) {
  const response = await fetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `请求失败（HTTP ${response.status}）`);
  return payload;
}

export default function Page() {
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [memories, setMemories] = useState<Record<string, string>>({});
  const [ready, setReady] = useState(false);
  const [identity, setIdentity] = useState<{ user_id: string; name: string } | null>(null);
  const [error, setError] = useState("");
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [memoryKey, setMemoryKey] = useState("");
  const [memoryValue, setMemoryValue] = useState("");
  const pending = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const transcript = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = transcript.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [messages]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    let active = true;
    async function restore() {
      try {
        const currentIdentity = await api('/api/identity');
        if (active) setIdentity(currentIdentity);
        let saved: string | null = null;
        try { saved = localStorage.getItem(SESSION_KEY); }
        catch { if (active) setError("浏览器禁止存储，本次会话无法在刷新后自动恢复。"); }
        if (saved) {
          const response = await fetch(`/api/sessions/${saved}`);
          if (response.status === 404 || response.status === 422) localStorage.removeItem(SESSION_KEY);
          else {
            if (!response.ok) throw new Error("无法恢复会话，请刷新页面重试。");
            const data = await response.json();
            if (active) { setSessionId(data.session_id); setMessages(data.messages); setMemories(data.memories); }
          }
        }
        if (active) setReady(true);
      } catch (error) {
        if (active) setError(error instanceof Error ? error.message : "无法恢复会话，请刷新重试。");
      }
    }
    void restore();
    return () => { active = false; };
  }, []);

  async function ensureSession() {
    if (sessionId) return sessionId;
    const session = await api("/api/sessions", { method: "POST" });
    setSessionId(session.session_id);
    try { localStorage.setItem(SESSION_KEY, session.session_id); }
    catch { setError("浏览器禁止存储，刷新后将无法自动恢复本次会话。"); }
    return session.session_id as string;
  }

  async function sendMessage() {
    const question = draft.trim();
    if (!question || question.length > 4000 || pending.current || !ready) return;
    pending.current = true;
    setIsLoading(true);
    setDraft("");
    setError("");
    const answerId = crypto.randomUUID();
    setMessages((previous) => [...previous,
      { id: crypto.randomUUID(), role: "user", content: question, status: "complete" },
      { id: answerId, role: "assistant", content: "", status: "pending" },
    ]);
    const requestController = new AbortController();
    controller.current = requestController;
    const finish = (content: string, status: Message["status"], tool_calls?: ToolExecution[], agent_events?: AgentEvent[], decision?: Message['decision']) => {
      setMessages((previous) => previous.map((item) => item.id === answerId ? { ...item, content, status, tool_calls, agent_events, decision } : item));
    };
    try {
      const id = await ensureSession();
      const response = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, session_id: id }), signal: requestController.signal,
      });
      const data: unknown = await response.json().catch(() => null);
      const payload = data && typeof data === "object" ? data as Record<string, unknown> : {};
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `请求失败（HTTP ${response.status}），请稍后重试。`);
      if (typeof payload.reply !== "string" || !payload.reply.trim()) throw new Error("模型没有返回有效回复，请重新发送问题。");
      finish(payload.reply, "complete", payload.tool_calls as ToolExecution[], payload.agent_events as AgentEvent[], payload.decision as Message['decision']);
      setMemories(payload.memories as Record<string, string>);
    } catch (error) {
      if (!requestController.signal.aborted) finish(error instanceof Error ? error.message : "请求失败，请稍后重试。", "error");
    } finally {
      pending.current = false;
      if (!requestController.signal.aborted) setIsLoading(false);
    }
  }

  async function clearChat() {
    if (pending.current || !ready) return;
    pending.current = true;
    setIsLoading(true);
    setError("");
    try {
      if (sessionId) await api(`/api/sessions/${sessionId}`, { method: "DELETE" });
      setSessionId(null); setMessages([]); setMemories({}); setDraft("");
      try { localStorage.removeItem(SESSION_KEY); } catch { /* Storage may be disabled. */ }
      input.current?.focus();
    } catch (error) { setError(error instanceof Error ? error.message : "清空失败"); }
    finally { pending.current = false; setIsLoading(false); }
  }

  async function changeMemory(key?: string) {
    if (pending.current || !ready) return;
    pending.current = true; setIsLoading(true); setError("");
    try {
      const id = await ensureSession();
      const data = await api(`/api/sessions/${id}/memories${key ? `/${encodeURIComponent(key)}` : ""}`, {
        method: key ? "DELETE" : "PUT",
        headers: { "Content-Type": "application/json" },
        ...(key ? {} : { body: JSON.stringify({ key: memoryKey, value: memoryValue }) }),
      });
      setMemories(data.memories);
      if (!key) { setMemoryKey(""); setMemoryValue(""); }
    } catch (error) { setError(error instanceof Error ? error.message : "记忆更新失败"); }
    finally { pending.current = false; setIsLoading(false); }
  }

  return (
    <main className="chat-layout">
      <aside className="sidebar" aria-label="对话概览">
        <div className="brand"><span className="brand-mark" aria-hidden="true">Q</span> 电商客服</div>
        <div className="sidebar-section" aria-label="当前用户身份">
          <p className="sidebar-label">当前用户（固定）</p>
          <p>{identity ? `${identity.name} · ${identity.user_id}` : '正在确认身份…'}</p>
          <span className="sidebar-caption">仅可查询该用户的订单，聊天中不能切换身份。</span>
        </div>
        <button className="new-chat" onClick={clearChat} disabled={isLoading || !ready || (!messages.length && !Object.keys(memories).length)}>＋ 清空对话与记忆</button>
        <div className="sidebar-section">
          <p className="sidebar-label">当前对话</p>
          <p className="conversation-title">{messages.find((item) => item.role === "user")?.content || "开始一段新对话"}</p>
          <span className="sidebar-caption">{messages.filter((item) => item.role === "user").length} 条提问 · 本地持久保存</span>
        </div>
        <div className="sidebar-bottom">本地 Qwen 模型<p>商品咨询 · 订单查询 · 售后解答。</p></div>
      </aside>
      <section className="chat-main" aria-label="电商客服对话">
        <header className="chat-header">
          <div><h1>电商客服 <span>本地演示</span></h1><p>商品咨询 · 物流查询 · 售后解答</p></div>
          <button className="clear-button" onClick={clearChat} disabled={isLoading || !ready || (!messages.length && !Object.keys(memories).length)}>清空对话与记忆</button>
          <button className="clear-button" onClick={() => setMemoryOpen(!memoryOpen)} aria-expanded={memoryOpen}>记忆（{Object.keys(memories).length}）</button>
        </header>
        {error && <p className="page-error" role="alert">{error}</p>}
        {!ready && !error && <p className="session-note">正在恢复会话…</p>}
        {memoryOpen && <section className="memory-panel" aria-label="会话记忆">
          <p>当前会话的长期记忆。也可以直接说“请记住…”或“忘记…”。</p>
          {Object.entries(memories).map(([key, value]) => <div className="memory-item" key={key}>
            <span><strong>{key}</strong>：{value}</span>
            <button className="clear-button" disabled={isLoading || !ready} onClick={() => changeMemory(key)}>删除</button>
          </div>)}
          {!Object.keys(memories).length && <p>暂无记忆</p>}
          <form onSubmit={(event) => { event.preventDefault(); void changeMemory(); }}>
            <input aria-label="记忆名称" placeholder="名称（相同名称会更新）" value={memoryKey} onChange={(event) => setMemoryKey(event.target.value)} maxLength={40} required />
            <input aria-label="记忆内容" placeholder="要记住的偏好或事实" value={memoryValue} onChange={(event) => setMemoryValue(event.target.value)} maxLength={200} required />
            <button className="clear-button" disabled={isLoading || !ready || !memoryKey.trim() || !memoryValue.trim()}>保存</button>
          </form>
        </section>}
        <CommercePanel disabled={isLoading || !ready} onSelect={(question) => { setDraft(question); input.current?.focus(); }} />
        <div className="transcript" ref={transcript}>
          {messages.length === 0 ? (
            <section className="welcome">
              <span className="welcome-mark" aria-hidden="true">Q</span>
              <h2>有什么可以帮你？</h2><p>咨询商品、查询订单，或了解退货退款政策。</p>
              <div className="suggestions">{suggestions.map((suggestion) => (
                <button key={suggestion} onClick={() => { setDraft(suggestion); input.current?.focus(); }}>{suggestion}<span aria-hidden="true">↗</span></button>
              ))}</div>
            </section>
          ) : (
            <div className="message-list" role="log" aria-label="聊天记录" aria-live="polite" aria-relevant="additions text">
              {messages.map((message) => <ChatMessage key={message.id} message={message} />)}
            </div>
          )}
        </div>
        <div className="composer-area">
          <ChatComposer draft={draft} onChange={setDraft} onSend={sendMessage} isLoading={isLoading || !ready} inputRef={input} />
          <p className="session-note">这是虚拟商店演示，不执行真实交易。会话保存在本机；清空会删除会话与记忆。</p>
        </div>
      </section>
    </main>
  );
}
