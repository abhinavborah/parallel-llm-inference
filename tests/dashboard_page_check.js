// Runs the dashboard page's own script against a small stand-in for the browser's DOM, feeds
// it the events the server sends, and checks what it draws and what it asks the server to do.
// Used by tests/test_dashboard.py when node is installed:
//   node dashboard_page_check.js PAGE.html STATE.json
// STATE.json is a real "state" event taken from a running dashboard.
"use strict";
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync(process.argv[2], "utf8");
const state = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];

const made = [];
class Node {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.className = "";
    this.dataset = {};
    this.attrs = {};
    this.listeners = {};
    this.title = "";
    this.value = "";
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.removed = false;
    this._text = "";
    made.push(this);
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  append(...nodes) { this.children.push(...nodes); }
  prepend(node) { this.children.unshift(node); }
  remove() { this.removed = true; }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  click() { return Promise.all((this.listeners.click || []).map((fn) => fn())); }
  find(test, out = []) {
    if (test(this)) out.push(this);
    this.children.forEach((c) => c.find && c.find(test, out));
    return out;
  }
}
class Text {
  constructor(value) { this.textContent = String(value); }
}

const ids = {};
for (const id of ["stages", "tiles", "prompt", "max", "requests", "message", "conn"]) {
  ids[id] = new Node("div");
}
ids.prompt.value = "The capital of France is";
ids.max.value = "16";
const modeButtons = ["one", "eight", "flood"].map((mode) => {
  const button = new Node("button");
  button.dataset.mode = mode;
  return button;
});

const sources = [];
class EventSource {
  constructor(url) { this.url = url; this.listeners = {}; this.readyState = 0; sources.push(this); }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  close() { this.readyState = 2; this.closed = true; }
  emit(type, data) { this.listeners[type]({ data: JSON.stringify(data) }); }
}
const fetched = [];
const timers = [];
const sandbox = {
  document: {
    getElementById: (id) => ids[id],
    createElement: (tag) => new Node(tag),
    createTextNode: (text) => new Text(text),
    querySelectorAll: () => modeButtons,
  },
  EventSource,
  fetch: async (path, options) => {
    fetched.push({ path, body: JSON.parse(options.body), headers: options.headers });
    return { ok: true, json: async () => ({ ok: true, ids: ["d1"], skipped: 0 }) };
  },
  console,
  setTimeout: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
  clearTimeout: () => {},
};
vm.runInNewContext(script, sandbox, { filename: "dashboard page script" });

const source = sources[0];
assert.strictEqual(source.url, "/events", "the page reads /events");
const stageCards = ids.stages.children;
assert.strictEqual(stageCards.length, 2, "one card per stage");
assert.strictEqual(ids.tiles.children.length, 8, "eight counters");

// ---- the state event the server sends is drawn
source.emit("state", state);
const badge = (n) => stageCards[n].find((x) => x.className === "badge")[0];
for (const stage of ["0", "1"]) {
  assert.strictEqual(badge(Number(stage)).textContent, state.stages[stage].state);
  const text = stageCards[Number(stage)].textContent;
  assert.ok(text.includes(String(state.stages[stage].epoch)), "the epoch is shown");
  assert.ok(text.includes(state.stages[stage].addr), "the address is shown");
  assert.ok(/MiB|GiB/.test(text), "free RAM is shown in units");
}
const tiles = ids.tiles.children.map((tile) => tile.children[0].textContent);
assert.strictEqual(tiles[0], `${state.running} / ${state.max_running}`, "running");
assert.strictEqual(tiles[1], `${state.queued} / ${state.max_queued}`, "queued");
assert.strictEqual(tiles[6], "0", "duplicate tokens");
assert.notStrictEqual(ids.tiles.children[6].dataset.bad, "true");

// ---- a dead stage and a duplicate are shown as bad
const dead = JSON.parse(JSON.stringify(state));
dead.stages["1"].state = "dead";
dead.duplicates = 2;
dead.discards = { duplicate: 1, unknown_request: 3 };
dead.stale = 4;
source.emit("state", dead);
assert.strictEqual(badge(1).textContent, "dead");
assert.strictEqual(badge(1).dataset.state, "dead");
assert.strictEqual(ids.tiles.children[6].dataset.bad, "true", "duplicates above 0 are flagged");
assert.ok(ids.tiles.children[5].title.includes("unknown_request: 3"), "stale reasons on hover");

// ---- a request's card fills with its tokens and ends
source.emit("start", { id: "d0", prompt: "an earlier request", max_tokens: 4 });
source.emit("start", { id: "d1", prompt: "The capital", max_tokens: 4 });
assert.strictEqual(ids.requests.children.length, 2);
source.emit("token", { id: "d1", step: 0, text: "Hello", duplicate: false });
source.emit("token", { id: "d1", step: 1, text: " <img src=x onerror=alert(1)>", duplicate: false });
source.emit("token", { id: "d1", step: 1, text: "!", duplicate: true });
const card = ids.requests.children[0];
assert.ok(card.textContent.includes("Hello <img src=x onerror=alert(1)>!"), "model text is text");
assert.ok(!ids.requests.children[1].textContent.includes("Hello"), "tokens go to their own card");
assert.strictEqual(made.filter((n) => n.tag === "img").length, 0, "no element made from text");
assert.strictEqual(card.find((x) => x.className === "dup").length, 1, "a duplicate is marked");
source.emit("end", { id: "d1", outcome: "done", tokens: 3 });
assert.strictEqual(card.find((x) => x.className === "badge")[0].textContent, "done");
source.emit("start", { id: "d2", prompt: "x", max_tokens: 4 });
source.emit("end", { id: "d2", outcome: "RESOURCE_EXHAUSTED", detail: "queue full", tokens: 0 });
const failed = ids.requests.children[0].find((x) => x.className === "badge")[0];
assert.strictEqual(failed.textContent, "RESOURCE_EXHAUSTED");
assert.strictEqual(failed.dataset.bad, "true");
source.emit("token", { id: "gone", step: 0, text: "x", duplicate: false }); // no card: ignored

// ---- the cards are capped
for (let i = 3; i < 40; i++) source.emit("start", { id: `d${i}`, prompt: "x", max_tokens: 1 });
assert.ok(ids.requests.children.filter((c) => !c.removed).length <= 24, "at most 24 cards");

// ---- a stream the server refuses is tried again later, not left dead
source.readyState = 2; // what a browser does after a 503 on /events
source.onerror();
assert.ok(ids.conn.textContent.includes("trying again"), "the page says it will try again");
assert.strictEqual(ids.conn.dataset.up, "false");
assert.ok(source.closed, "the dead stream is closed");
assert.strictEqual(timers.length, 1);
assert.strictEqual(timers[0].ms, 5000);
timers[0].fn();
assert.strictEqual(sources.length, 2, "a new stream is opened");
sources[1].onopen();
assert.strictEqual(ids.conn.textContent, "live");
sources[1].emit("state", state);
assert.strictEqual(badge(0).textContent, state.stages["0"].state, "and it is drawn from");
sources[1].readyState = 0; // a dropped stream the browser retries itself
sources[1].onerror();
assert.strictEqual(ids.conn.textContent, "disconnected, retrying");
assert.strictEqual(timers.length, 1, "no second timer while the browser retries");

// ---- the buttons ask the server for what they say
(async () => {
  await modeButtons[1].click();
  assert.deepStrictEqual(fetched.at(-1).body, {
    prompt: "The capital of France is", max_tokens: 16, mode: "eight",
  });
  assert.strictEqual(fetched.at(-1).path, "/generate");
  assert.strictEqual(fetched.at(-1).headers["Content-Type"], "application/json");
  await modeButtons[2].click();
  assert.strictEqual(fetched.at(-1).body.mode, "flood");
  const button = (stage, label) =>
    stageCards[stage].find((x) => x.tag === "button" && x.textContent === label)[0];
  await button(1, "Kill").click();
  assert.deepStrictEqual(fetched.at(-1).body, { stage: 1, action: "kill" });
  assert.strictEqual(fetched.at(-1).path, "/fault");
  await button(0, "Freeze").click();
  assert.deepStrictEqual(fetched.at(-1).body, { stage: 0, action: "suspend" });
  await button(0, "Unfreeze").click();
  assert.deepStrictEqual(fetched.at(-1).body, { stage: 0, action: "resume" });
  assert.ok(ids.message.textContent.includes("Unfreeze stage 0"), "the result is said");
  console.log("page script ok");
})().catch((err) => { console.error(err); process.exit(1); });
