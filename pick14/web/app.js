/**
 * Pick14 web client — uses same-origin `/sessions` API.
 */

const SUIT_NAMES = ["CLUB", "DIAMOND", "HEART", "BLADE"];

function rankLabel(rank) {
  if (rank === 1) return "A";
  if (rank === 11) return "J";
  if (rank === 12) return "Q";
  if (rank === 13) return "K";
  if (rank >= 2 && rank <= 10) return String(rank);
  return "?";
}

function formatCard(c) {
  if (c.joker) {
    return c.red ? "RED JOKER" : "BLACK JOKER";
  }
  const suit = SUIT_NAMES[c.suit] || "?";
  return `${rankLabel(c.rank)} of ${suit}`;
}

function scoreValue(c) {
  if (c.joker) return 5;
  const m = { 0: 1, 1: 2, 2: 4, 3: 3 };
  return m[c.suit] ?? 0;
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

let sessionId = null;

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

function renderCards(ul, cards) {
  ul.innerHTML = "";
  for (const c of cards) {
    const li = document.createElement("li");
    li.className = "card-pill";
    li.textContent = formatCard(c);
    ul.appendChild(li);
  }
}

function renderScores(state) {
  const ul = document.getElementById("scoreList");
  ul.innerHTML = "";
  const n = state.hands.length;
  for (let i = 0; i < n; i++) {
    const li = document.createElement("li");
    const pile = state.score_piles[i] || [];
    const pts = totalScorePile(pile);
    const [sets, rem] = setsAndPoints(pts);
    const label = i === 0 ? "You" : `Player ${i}`;
    li.textContent = `${label}: ${pts} pts (${sets} sets + ${rem})`;
    ul.appendChild(li);
  }
}

function renderMoves(view) {
  const moves = view.legal_moves || [];
  const matchGroup = document.getElementById("matchGroup");
  const playGroup = document.getElementById("playGroup");
  const matchBtns = document.getElementById("matchButtons");
  const playBtns = document.getElementById("playButtons");
  matchBtns.innerHTML = "";
  playBtns.innerHTML = "";

  const matches = moves.filter((m) => m.kind === "match");
  const plays = moves.filter((m) => m.kind === "play");

  if (matches.length && !view.state.must_play_only) {
    matchGroup.hidden = false;
    const skip = document.createElement("button");
    skip.type = "button";
    skip.className = "btn btn--move";
    skip.textContent = "Skip matching — choose a play below";
    skip.addEventListener("click", () => {
      playGroup.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    matchBtns.appendChild(skip);

    const hand = view.state.hands[0] || [];
    const pub = view.state.public || [];
    matches.forEach((m, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn--move";
      const pubCard = pub[m.public_index];
      const parts = m.hand_indices.map((idx) => formatCard(hand[idx])).join(", ");
      const cap = m.hand_indices.reduce((s, idx) => s + scoreValue(hand[idx]), scoreValue(pubCard));
      btn.textContent = `[${i + 1}] (${cap} pts) pick ${formatCard(pubCard)} by ${parts}`;
      btn.addEventListener("click", () => submitMove({ kind: "match", public_index: m.public_index, hand_indices: m.hand_indices }));
      matchBtns.appendChild(btn);
    });
  } else {
    matchGroup.hidden = true;
  }

  if (plays.length) {
    playGroup.hidden = false;
    const hand = view.state.hands[0] || [];
    plays.forEach((m, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn--move";
      const c = hand[m.hand_index];
      btn.textContent = `[${i}] (${scoreValue(c)} pts) play ${formatCard(c)}`;
      btn.addEventListener("click", () => submitMove({ kind: "play", hand_index: m.hand_index }));
      playBtns.appendChild(btn);
    });
  } else {
    playGroup.hidden = true;
  }
}

async function submitMove(payload) {
  if (!sessionId) return;
  setStatus("…");
  try {
    const sid = sessionId;
    const view = await api("POST", `/sessions/${sid}/moves/human`, payload);
    applyView(view);
    setStatus("");
  } catch (e) {
    setStatus(e.message || String(e), true);
  }
}

function applyView(view) {
  const st = view.state;
  document.getElementById("game").hidden = false;
  document.getElementById("finished").hidden = true;

  renderCards(document.getElementById("publicList"), st.public || []);
  renderCards(document.getElementById("handList"), st.hands[0] || []);
  renderScores(st);

  document.getElementById("mustPlay").hidden = !st.must_play_only;

  if (view.finished) {
    document.getElementById("game").hidden = true;
    document.getElementById("finished").hidden = false;
    const fs = document.getElementById("finalScores");
    fs.innerHTML = "";
    const n = st.hands.length;
    for (let i = 0; i < n; i++) {
      const pts = totalScorePile(st.score_piles[i] || []);
      const [sets, rem] = setsAndPoints(pts);
      const li = document.createElement("li");
      const label = i === 0 ? "You" : `Player ${i}`;
      li.textContent = `${label}: ${pts} points (${sets} sets and ${rem} points)`;
      fs.appendChild(li);
    }
    return;
  }

  if (view.current_player !== 0) {
    setStatus("Waiting for bots…");
    pollUntilHuman();
    return;
  }

  renderMoves(view);
}

let pollTimer = null;

function pollUntilHuman() {
  if (pollTimer) clearInterval(pollTimer);
  const sid = sessionId;
  pollTimer = setInterval(async () => {
    try {
      const view = await api("GET", `/sessions/${sid}`);
      if (view.finished || view.current_player === 0) {
        clearInterval(pollTimer);
        pollTimer = null;
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
  if (seedRaw !== "") {
    body.seed = parseInt(seedRaw, 10);
  }
  setStatus("Starting…");
  try {
    const data = await api("POST", "/sessions", body);
    sessionId = data.session_id;
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
    setStatus(
      view.remaining_completed_segments != null
        ? `${view.remaining_completed_segments} older segment(s) still reversible`
        : ""
    );
    applyView(view);
  } catch (e) {
    setStatus(e.message || String(e), true);
  }
}

async function refresh() {
  if (!sessionId) return;
  try {
    const view = await api("GET", `/sessions/${sessionId}`);
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
