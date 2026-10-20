import { useEffect, type RefObject } from "react";

type Props = {
  draft: string;
  onChange: (value: string) => void;
  onSend: () => void;
  isLoading: boolean;
  inputRef: RefObject<HTMLTextAreaElement | null>;
};

export function ChatComposer({ draft, onChange, onSend, isLoading, inputRef }: Props) {
  useEffect(() => {
    const input = inputRef.current;
    if (input) {
      input.style.height = "auto";
      input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
    }
  }, [draft, inputRef]);
  return (
    <form className="composer" onSubmit={(event) => { event.preventDefault(); onSend(); }}>
      <label className="sr-only" htmlFor="question">输入问题</label>
      <textarea id="question" ref={inputRef} value={draft} rows={1} maxLength={4000}
        placeholder="向 Qwen 提问…" aria-describedby="composer-help"
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) {
            event.preventDefault(); onSend();
          }
        }}
      />
      <div className="composer-toolbar">
        <span id="composer-help" className="composer-help"><span className="keyboard-hint">Enter 发送 · Shift + Enter 换行</span><span>{draft.length} / 4000</span></span>
        <button className="send-button" type="submit" disabled={!draft.trim() || isLoading} aria-label={isLoading ? "等待模型回复" : "发送问题"}>
          <span aria-hidden="true">{isLoading ? "…" : "↑"}</span>
        </button>
      </div>
    </form>
  );
}
