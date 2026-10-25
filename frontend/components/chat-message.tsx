import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const showDeveloperDetails = process.env.NEXT_PUBLIC_SHOW_DEV_DETAILS !== "false";

export type ToolExecution = {
  step: number;
  name: string;
  arguments: string;
  result: { ok: boolean; result?: unknown; error?: string };
  agent?: string;
};

export type AgentEvent = {
  agent: string;
  task: string;
  status: string;
  tool_count: number;
  execution_state?: string | null;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "pending" | "complete" | "error";
  tool_calls?: ToolExecution[];
  agent_events?: AgentEvent[];
  decision?: { scope?: { status: string; source?: string; reason?: string } };
};

const agentLabels: Record<string, string> = {
  supervisor: "客服协调", orders: "订单客服", product: "商品顾问",
};
const eventLabels: Record<string, string> = {
  completed: "已返回", partial: "部分完成", failed: "遇到问题", limit: "达到执行上限",
};

const toolLabels: Record<string, string> = {
  list_orders: "我的订单", order_detail: "订单详情", catalog_search: "商品价格库存", knowledge_search: "商品知识检索",
  calculator: "计算器", current_time: "时间查询",
  memory_set: "保存记忆", memory_list: "读取记忆", memory_delete: "删除记忆",
};

const linkPattern = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)|(https?:\/\/[^\s<>"']+)/g;

function splitReply(content: string): { reply: string; citations: string[] } {
  const internalSource = `(?:KB-[a-zA-Z0-9]+|${Object.keys(toolLabels).join("|")})`;
  const citations = new RegExp(`\\[${internalSource}(?:[\\s,，;；]+${internalSource})*\\](?!\\()`, "g");
  return {
    reply: content.replace(citations, "").replace(/^[ \t]+$/gm, "").trim(),
    citations: [...new Set(content.match(citations) || [])],
  };
}

function linkify(content: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  for (const match of content.matchAll(linkPattern)) {
    const index = match.index ?? 0;
    if (index > cursor) nodes.push(content.slice(cursor, index));
    const rawHref = match[2] || match[3];
    const href = rawHref.replace(/[.,!?;:，。！？；：]+$/u, "");
    const label = match[1] || href;
    nodes.push(<a key={index} href={href} target="_blank" rel="noopener noreferrer">{label}</a>);
    const suffix = rawHref.slice(href.length);
    if (suffix) nodes.push(suffix);
    cursor = index + match[0].length;
  }
  if (cursor < content.length) nodes.push(content.slice(cursor));
  return nodes;
}

export function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const { reply, citations } = splitReply(message.content);
  return (
    <article className={`message message-${message.role}`} aria-label={isUser ? "你的提问" : "客服的回复"}>
      {!isUser && <span className="assistant-avatar" aria-hidden="true">Q</span>}
      <div className="message-body">
        <span className="message-author">{isUser ? "你" : "电商客服"}</span>
        {message.status === "pending" ? (
          <p className="thinking"><span className="thinking-dot" aria-hidden="true" />正在处理…</p>
        ) : (
          <div className={`message-content${message.status === "error" ? " message-error" : ""}`}>
            {message.status === "error" && <strong>本次回复失败</strong>}
            {isUser || message.status === "error" ? <p>{linkify(message.content)}</p> : (
              <div className="markdown-reply">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  skipHtml
                  components={{
                    a: ({ children, href, title }) => <a href={href} title={title} target="_blank" rel="noopener noreferrer">{children}</a>,
                    table: ({ children }) => <div className="markdown-table"><table>{children}</table></div>,
                  }}
                >{reply}</ReactMarkdown>
              </div>
            )}
            {showDeveloperDetails && !isUser && citations.length > 0 && <details className="tool-trace">
              <summary>来源引用</summary>
              {citations.map((citation) => <p key={citation}>{citation}</p>)}
            </details>}
            {showDeveloperDetails && !isUser && <details className="tool-trace">
              <summary>开发者诊断</summary>
              <pre>{message.content}</pre>
              {message.decision && <pre>{JSON.stringify(message.decision, null, 2)}</pre>}
            </details>}
            {showDeveloperDetails && !!message.agent_events?.some((event) => event.agent !== "supervisor") && <details className="tool-trace">
              <summary>客服协作记录</summary>
              {message.agent_events.map((event, index) => <div key={index}>
                <strong>{agentLabels[event.agent] || event.agent} · {eventLabels[event.status] || event.status}</strong>
                <p>{event.task}{event.tool_count > 0 ? ` · ${event.tool_count} 次工具调用` : ""}</p>
              </div>)}
            </details>}
            {showDeveloperDetails && !!message.tool_calls?.length && <details className="tool-trace">
              <summary>工具执行记录（{message.tool_calls.length}）</summary>
              {message.tool_calls.map((tool, index) => <div key={index}>
                <strong>{agentLabels[tool.agent || "supervisor"] || tool.agent} · 第 {tool.step} 轮 · {toolLabels[tool.name] || tool.name} · {tool.result.ok ? "成功" : "失败"}</strong>
                <pre>{tool.arguments}</pre>
                <pre>{JSON.stringify(tool.result, null, 2)}</pre>
              </div>)}
            </details>}
          </div>
        )}
      </div>
    </article>
  );
}

