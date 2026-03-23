/**
 * Pick14 web client — select cards, then confirm one action.
 */

const SUIT_NAMES = ["CLUB", "DIAMOND", "HEART", "BLADE"];
const SUIT_BASE = [0x1f0d1, 0x1f0c1, 0x1f0b1, 0x1f0a1];
const JOKER_RED = 0x1f0bf;
const JOKER_BLACK = 0x1f0cf;

let sessionId = null;
let currentView = null;
const selectedHand = new Set();
let selectedPublic = null;
let pollTimer = null;

function rankOffset(rank) {
  if (rank === 1) return 0;
  if (rank >= 2 && rank <= 9) return rank - 1;
  if (rank === 10) return 9;
  if (rank === 11) return 10;
  if (rank === 12) return 12;
  if (rank === 13) return 13;
  return 0;
}

function rankLabel(rank) {
  if (rank === 1) return "A";
  if (rank === 11) return "J";
  if (rank === 12) return "Q";
  if (rank === 13) return "K";
  if (rank >= 2 && rank <= 10) return String(rank);
  return "?";
}

function formatCardText(c) {
  if (c.joker) return c.red ? "RED JOKER" : "BLACK JOKER";
  return `${rankLabel(c.rank)} of ${SUIT_NAMES[c.suit] || "?"}`;
}

// Kept for tests that look for function name
function formatCard(c) {
  return formatCardText(c);
}

function cardGlyph(c) {
  if (c.joker) return String.fromCodePoint(c.red ? JOKER_RED : JOKER_BLACK);
  return String.fromCodePoint(SUIT_BASE[c.suit] + rankOffset(c.rank));
}

function toneClass(c) {
  if (c.joker) return c.red ? "card-tile--joker-red" : "card-tile--joker-black";
  return c.suit === 1 || c.suit === 2 ? "card-tile--red" : "card-tile--black";
}

function scoreValue(c) {
  if (c.joker) return 5;
  return { 0: 1, 1: 2, 2: 4, 3: 3 }[c.suit] ?? 0;
}

function totalScorePile(cards) {
  return cards.reduce((s, c) => s + scoreValue(c), 0);
}

function setsAndPoints(points) {
  return [Math.floor(points / 4), points % 4];
}

function setStatus(msg, isError) {
  const el = document.getElementById("status");
  el.textContent = msg || "";
  el.classList.toggle("err", Boolean(isError));
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new Error(`Bad JSON (${res.status})`);
  }
  if (!res.ok) {
    const detail = data?.detail ?? res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function cardButton(c, { selectable, selected, publicSelected, onClick }) {
  const li = document.createElement("li");
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "card-btn";
  if (selectable) btn.classList.add("is-selectable");
  if (selected) btn.classList.add("is-selected");
  if (publicSelected) btn.classList.add("is-public-selected");
  btn.disabled = !selectable;
  if (onClick) btn.addEventListener("click", onClick);

  const face = document.createElement("div");
  face.className = `card-tile ${toneClass(c)}`;
  face.setAttribute("role", "img");
  face.setAttribute("aria-label", formatCardText(c));
  face.title = formatCardText(c);

  const fallback = document.createElement("span");
  fallback.className = "card-tile__fallback";
  fallback.textContent = formatCardText(c);

  const glyph = document.createElement("span");
  glyph.className = "card-tile__glyph";
  glyph.textContent = cardGlyph(c);
  glyph.setAttribute("aria-hidden", "true");

  face.appendChild(fallback);
  face.appendChild(glyph);
  btn.appendChild(face);
  li.appendChild(btn);
  return li;
}

function renderCardLists() {
  const view = currentView;
  if (!view) return;
  const state = view.state;
  const hand = state.hands[0] || [];
  const pub = state.public || [];
  const active = view.current_player === 0 && !view.finished;

  const handList = document.getElementById("handList");
  handList.innerHTML = "";
  hand.forEach((c, i) => {
    handList.appendChild(
      cardButton(c, {
        selectable: active,
        selected: selectedHand.has(i),
        onClick: () => {
          if (selectedHand.has(i)) selectedHand.delete(i);
          else selectedHand.add(i);
          syncActionPanel();
          renderCardLists();
        },
      })
    );
  });

  const publicList = document.getElementById("publicList");
  publicList.innerHTML = "";
  pub.forEach((c, i) => {
    publicList.appendChild(
      cardButton(c, {
        selectable: active && !state.must_play_only,
        publicSelected: selectedPublic === i,
        onClick: () => {
          selectedPublic = selectedPublic === i ? null : i;
          syncActionPanel();
          renderCardLists();
        },
      })
    );
  });
}

function renderScores(state) {
  const ul = document.getElementById("scoreList");
  ul.innerHTML = "";
  for (let i = 0; i < state.hands.length; i++) {
    const li = document.createElement("li");
    const pts = totalScorePile(state.score_piles[i] || []);
    const [sets, rem] = setsAndPoints(pts);
    const label = i === 0 ? "You" : `Player ${i}`;
    li.textContent = `${label}: ${pts} pts (${sets} sets + ${rem})`;
    ul.appendChild(li);
  }
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function selectedAction() {
  const view = currentView;
  if (!view || view.current_player !== 0 || view.finished) return null;
  const state = view.state;
  const legal = view.legal_moves || [];
  const hand = state.hands[0] || [];
  const pub = state.public || [];
  const handIdx = [...selectedHand].sort((a, b) => a - b);

  if (selectedPublic != null) {
    if (state.must_play_only) {
      return { valid: false, summary: "Cannot match now; this turn requires a play to public." };
    }
    if (handIdx.length === 0) return { valid: false, summary: "Select one or more hand cards for match." };
    const move = legal.find(
      (m) => m.kind === "match" && m.public_index === selectedPublic && arraysEqual((m.hand_indices || []).slice().sort((a,b)=>a-b), handIdx)
    );
    if (!move) return { valid: false, summary: "Selected set is not a legal match (sum must be 14)." };
    const cap = handIdx.reduce((s, i) => s + scoreValue(hand[i]), scoreValue(pub[selectedPublic]));
    const cards = handIdx.map((i) => formatCardText(hand[i])).join(", ");
    return {
      valid: true,
      payload: { kind: "match", public_index: selectedPublic, hand_indices: handIdx },
      summary: `(${cap} pts) pick ${formatCardText(pub[selectedPublic])} by ${cards}`,
    };
  }

  if (handIdx.length === 1) {
    const idx = handIdx[0];
    const move = legal.find((m) => m.kind === "play" && m.hand_index === idx);
    if (!move) return { valid: false, summary: "That card cannot be played now." };
    const pts = scoreValue(hand[idx]);
    return {
      valid: true,
      payload: { kind: "play", hand_index: idx },
      summary: `(${pts} pts) play ${formatCardText(hand[idx])}`,
    };
  }

  if (handIdx.length > 1) return { valid: false, summary: "To play, select exactly one hand card." };
  return { valid: false, summary: "No action selected." };
}

function syncActionPanel() {
  const action = selectedAction();
  const btn = document.getElementById("btnConfirmAction");
  const info = document.getElementById("actionInfo");
  if (!action) {
    btn.disabled = true;
    info.textContent = "No action selected.";
    return;
  }
  btn.disabled = !action.valid;
  info.textContent = action.summary;
}

async function submitSelectedAction() {
  const action = selectedAction();
  if (!action || !action.valid) return;
  setStatus("…");
  try {
    const view = await api("POST", `/sessions/${sessionId}/moves/human`, action.payload);
    selectedHand.clear();
    selectedPublic = null;
    applyView(view);
    setStatus("");
  } catch (e) {
    setStatus(e.message || String(e), true);
  }
}

function applyView(view) {
  currentView = view;
  const st = view.state;
  document.getElementById("game").hidden = false;
  document.getElementById("finished").hidden = true;

  renderCardLists();
  renderScores(st);
  syncActionPanel();

  document.getElementById("mustPlay").hidden = !st.must_play_only;

  if (view.finished) {
    document.getElementById("game").hidden = true;
    document.getElementById("finished").hidden = false;
    const fs = document.getElementById("finalScores");
    fs.innerHTML = "";
    for (let i = 0; i < st.hands.length; i++) {
      const pts = totalScorePile(st.score_piles[i] || []);
      const [sets, rem] = setsAndPoints(pts);
      const li = document.createElement("li");
      li.textContent = `${i === 0 ? "You" : `Player ${i}`}: ${pts} points (${sets} sets and ${rem} points)`;
      fs.appendChild(li);
    }
    return;
  }

  if (view.current_player !== 0) {
    setStatus("Waiting for bots…");
    pollUntilHuman();
    return;
  }
}

function pollUntilHuman() {
  if (pollTimer) clearInterval(pollTimer);
  const sid = sessionId;
  pollTimer = setInterval(async () => {
    try {
      const view = await api("GET", `/sessions/${sid}`);
      if (view.finished || view.current_player === 0) {
        clearInterval(pollTimer);
        pollTimer = null;
        selectedHand.clear();
        selectedPublic = null;
        applyView(view);
        setStatus("");
      }
    } catch {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }, 400);
}

async function newGame() {
  const n = Math.min(8, Math.max(2, parseInt(document.getElementById("numPlayers").value, 10) || 2));
  document.getElementById("numPlayers").value = String(n);
  const seedRaw = document.getElementById("seed").value.trim();
  const body = { num_players: n };
  if (seedRaw !== "") body.seed = parseInt(seedRaw, 10);
  setStatus("Starting…");
  try {
    const data = await api("POST", "/sessions", body);
    sessionId = data.session_id;
    selectedHand.clear();
    selectedPublic = null;
    setStatus("");
    applyView(data);
  } catch (e) {
    setStatus(e.message || String(e), true);
  }
}

async function regret() {
  if (!sessionId) return;
  setStatus("…");
  try {
    const view = await api("POST", `/sessions/${sessionId}/regret`);
    selectedHand.clear();
    selectedPublic = null;
    setStatus(view.remaining_completed_segments != null ? `${view.remaining_completed_segments} older segment(s) still reversible` : "");
    applyView(view);
  } catch (e) {
    setStatus(e.message || String(e), true);
  }
}

async function refresh() {
  if (!sessionId) return;
  try {
    const view = await api("GET", `/sessions/${sessionId}`);
    selectedHand.clear();
    selectedPublic = null;
    applyView(view);
    setStatus("");
  } catch (e) {
    setStatus(e.message || String(e), true);
  }
}

document.getElementById("btnNew").addEventListener("click", newGame);
document.getElementById("btnAgain").addEventListener("click", newGame);
document.getElementById("btnRegret").addEventListener("click", regret);
document.getElementById("btnRefresh").addEventListener("click", refresh);
document.getElementById("btnConfirmAction").addEventListener("click", submitSelectedAction);
