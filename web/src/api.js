// SSE поверх POST: EventSource умеет только GET, поэтому читаем поток
// вручную через fetch + ReadableStream.
//
// Важное про разбор: кадр SSE заканчивается пустой строкой, но кадр может
// прийти разорванным между двумя chunk'ами. Поэтому держим буфер и режем
// только по завершённым кадрам, иначе на длинных ответах будут потери.

function parseFrame(frame) {
  let event = "message";
  const data = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  if (!data.length) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    return null;
  }
}

export async function streamAsk(q, handlers, signal) {
  const resp = await fetch("/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q }),
    signal,
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let cut;
    while ((cut = buf.indexOf("\n\n")) !== -1) {
      const parsed = parseFrame(buf.slice(0, cut));
      buf = buf.slice(cut + 2);
      if (!parsed) continue;
      const fn = handlers[parsed.event];
      if (fn) fn(parsed.data);
    }
  }
}
