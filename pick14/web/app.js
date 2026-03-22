/**
 * Pick14 web UI — Unicode playing-card codepoints (emoji-style glyphs) + fetch API.
 * @see https://en.wikipedia.org/wiki/Playing_cards_in_Unicode
 */

const SUIT_NAMES = ["CLUB", "DIAMOND", "HEART", "BLADE"];

/** Per-suit Ace code point (Unicode playing cards block). */
const SUIT_BASE = [0x1f0d1, 0x1f0c1, 0x1f0b1, 0x1f0a1];

const JOKER_RED = 0x1f0bf;
const JOKER_BLACK = 0x1f0cf;

function rankOffset(rank) {
  if (rank === 1) return 0;
  if (rank >= 2 && rank <= 9) return rank - 1;
  if (rank === 10) return 9;
  if (rank === 11) return 10;
  if (rank === 12) return 12;
  if (rank === 13) return 13;
  return 0;
}

function cardGlyph(c) {
  if (c.joker) {
    return String.fromCodePoint(c.red ? JOKER_RED : JOKER_BLACK);
  }
  const base = SUIT_BASE[c.suit];
  return String.fromCodePoint(base + rankOffset(c.rank));
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

/** CSS hook: red = diamond+heart, black = club+spade (BLADE). */
function cardToneClass(c) {
  if (c.joker) return c.red ? "card-tone--joker-red" : "card-tone--joker-black";
  if (c.suit === 1 || c.suit === 2) return "card-tone--red";
  return "card-tone--black";
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

/**
 * @param {object} c card dict from API
 * @param {{ small?: boolean, dealDelay?: number|null, extraClass?: string }} opts
 */
function createCardVisual(c, opts = {}) {
  const li = document.createElement("li");
  li.className = "card-visual-wrap";
  const face = document.createElement("div");
  face.className =
    "card-visual" + (opts.small ? " card-visual--sm" : "") + " " + cardToneClass(c);
  if (opts.dealDelay != null && opts.dealDelay >= 0) {
    face.classList.add("card-visual--deal");
    face.style.setProperty("--deal-delay", `${opts.dealDelay}ms`);
  }
  if (opts.extraClass) face.classList.add(opts.extraClass);

  const fb = document.createElement("span");
  fb.className = "card-visual__fallback";
  fb.textContent = formatCardText(c);
  fb.setAttribute("aria-hidden", "true");

  const glyph = document.createElement("span");
  glyph.className = "card-visual__glyph";
  glyph.textContent = cardGlyph(c);
  glyph.setAttribute("aria-hidden", "true");

  /* Fallback first in DOM so the glyph paints on top (was reversed → text covered the card character). */
  face.appendChild(fb);
  face.appendChild(glyph);
  face.title = formatCardText(c);
  face.setAttribute("role", "img");
  face.setAttribute("aria-label", formatCardText(c));
  li.appendChild(face);
  return li;
}

function cardFaceElement(c, opts = {}) {
  const wrap = createCardVisual(c, opts);
  return wrap.firstElementChild;
}

function miniCardRow(cards) {
  const span = document.createElement("span");
  span.className = "move-mini-cards";
  for (const c of cards) {
    span.appendChild(cardFaceElement(c, { small: true }));
  }
  return span;
}

let sessionId = null;
let dealAnimNext = false;
let prevScoreLens = null;
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

function renderCardRow(ul, cards, { deal } = {}) {
  ul.innerHTML = "";
  cards.forEach((c, i) => {
    const delay = deal ? Math.min(i * 55, 800) : null;
    ul.appendChild(createCardVisual(c, { dealDelay: delay }));
  });
}

function renderScores(state, { bumpPlayers } = {}) {
  const ul = document.getElementById("scoreBoard");
  const n = state.hands.length;
  const lens = state.score_piles.map((p) => p.length);

  ul.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const li = document.createElement("li");
    const pile = state.score_piles[i] || [];
    const pts = totalScorePile(pile);
    const [sets, rem] = setsAndPoints(pts);
    const label = i === 0 ? "You" : `Player ${i}`;

    li.innerHTML = `
      <div class="score-line">
        <span class="score-line__who">${label}</span>
        <span class="score-line__total">${pts}</span>
        <span class="score-line__meta">${sets} sets + ${rem} (4 pts = 1 set)</span>
      </div>
      <div class="score-cheat"></div>
    `;
    const cheat = li.querySelector(".score-cheat");
    for (const c of pile) {
      cheat.appendChild(cardFaceElement(c, { small: true }));
    }
    if (bumpPlayers && bumpPlayers.includes(i)) {
      li.classList.add("score-bump");
      setTimeout(() => li.classList.remove("score-bump"), 800);
    }
    ul.appendChild(li);
  }

  prevScoreLens = lens;
}

function scoreBumpIndices(state) {
  const lens = state.score_piles.map((p) => p.length);
  if (!prevScoreLens || prevScoreLens.length !== lens.length) return [];
  const out = [];
  for (let i = 0; i < lens.length; i++) {
    if (lens[i] > prevScoreLens[i]) out.push(i);
  }
  return out;
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
  const hand = view.state.hands[0] || [];
  const pub = view.state.public || [];

  if (matches.length && !view.state.must_play_only) {
    matchGroup.hidden = false;
    const skip = document.createElement("button");
    skip.type = "button";
    skip.className = "btn btn--move";
    skip.textContent = "Skip matching — scroll to plays";
    skip.addEventListener("click", () => playGroup.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    matchBtns.appendChild(skip);

    matches.forEach((m, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn--move";
      const pubCard = pub[m.public_index];
      const handCards = m.hand_indices.map((idx) => hand[idx]);
      const cap = handCards.reduce((s, c) => s + scoreValue(c), 0) + scoreValue(pubCard);
      const label = document.createElement("span");
      label.textContent = `[${i + 1}] (${cap} pts) pick `;
      btn.appendChild(label);
      btn.appendChild(miniCardRow([pubCard]));
      const mid = document.createElement("span");
      mid.textContent = " using ";
      btn.appendChild(mid);
      btn.appendChild(miniCardRow(handCards));
      btn.addEventListener("click", () => {
        pulsePublicAndHand(m.public_index, m.hand_indices);
        setTimeout(() => {
          submitMove({ kind: "match", public_index: m.public_index, hand_indices: m.hand_indices });
        }, 380);
      });
      matchBtns.appendChild(btn);
    });
  } else {
    matchGroup.hidden = true;
  }

  if (plays.length) {
    playGroup.hidden = false;
    plays.forEach((m, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn--move";
      const c = hand[m.hand_index];
      const lab = document.createElement("span");
      lab.textContent = `[${i}] (${scoreValue(c)} pts) play `;
      btn.appendChild(lab);
      btn.appendChild(miniCardRow([c]));
      btn.addEventListener("click", () => submitMove({ kind: "play", hand_index: m.hand_index }));
      playBtns.appendChild(btn);
    });
  } else {
    playGroup.hidden = true;
  }
}

function pulsePublicAndHand(pubIndex, handIndices) {
  const pubLis = document.querySelectorAll("#publicList .card-visual");
  const handLis = document.querySelectorAll("#handList .card-visual");
  if (pubLis[pubIndex]) pubLis[pubIndex].classList.add("card-visual--match-pulse");
  for (const idx of handIndices) {
    if (handLis[idx]) handLis[idx].classList.add("card-visual--match-pulse");
  }
  setTimeout(() => {
    document.querySelectorAll(".card-visual--match-pulse").forEach((el) => el.classList.remove("card-visual--match-pulse"));
  }, 600);
}

async function submitMove(payload) {
  if (!sessionId) return;
  setStatus("…");
  try {
    const view = await api("POST", `/sessions/${sessionId}/moves/human`, payload);
    applyView(view, { afterMove: true });
    setStatus("");
  } catch (e) {
    setStatus(e.message || String(e), true);
  }
}

function triggerDeckShuffleAnim() {
  const stage = document.getElementById("deckStage");
  const backs = stage.querySelectorAll(".deck-card-back");
  stage.hidden = false;
  stage.classList.remove("deck--shuffling");
  void stage.offsetWidth;
  stage.classList.add("deck--shuffling");
  backs.forEach((el, i) => {
    el.style.setProperty("--tx", `${i * 3}px`);
    el.style.setProperty("--ty", `${i * 2}px`);
    el.style.setProperty("--rot", `${-4 + i * 3}deg`);
  });
  setTimeout(() => stage.classList.remove("deck--shuffling"), 1000);
}

function applyView(view, { afterMove = false, isNewGame = false } = {}) {
  const st = view.state;
  document.getElementById("game").hidden = false;
  document.getElementById("finished").hidden = true;

  const doDeal = isNewGame || dealAnimNext;
  dealAnimNext = false;

  const bumps = afterMove ? scoreBumpIndices(st) : [];

  renderCardRow(document.getElementById("publicList"), st.public || [], { deal: doDeal });
  renderCardRow(document.getElementById("handList"), st.hands[0] || [], { deal: doDeal });
  renderScores(st, { bumpPlayers: bumps });

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
    setStatus("Bots thinking…");
    pollUntilHuman();
    return;
  }

  renderMoves(view);
}

let pollTimer = null;

function pollUntilHuman() {
  if (pollTimer) clearInterval(pollTimer);
  const sid = sessionId;
  let n = 0;
  pollTimer = setInterval(async () => {
    n += 1;
    if (n > 50) {
      clearInterval(pollTimer);
      pollTimer = null;
      setStatus("Stuck waiting — try Refresh.", true);
      return;
    }
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
  if (seedRaw !== "") body.seed = parseInt(seedRaw, 10);

  dealAnimNext = true;
  prevScoreLens = null;
  triggerDeckShuffleAnim();

  setStatus("Dealing…");
  try {
    const data = await api("POST", "/sessions", body);
    sessionId = data.session_id;
    setStatus("");
    applyView(data, { isNewGame: true });
  } catch (e) {
    setStatus(e.message || String(e), true);
  }
}

async function regret() {
  if (!sessionId) return;
  setStatus("…");
  try {
    const view = await api("POST", `/sessions/${sessionId}/regret`);
    const rem = view.remaining_completed_segments;
    setStatus(rem != null ? `${rem} older segment(s) still reversible` : "");
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

document.getElementById("cheatToggle").addEventListener("change", (e) => {
  document.getElementById("scoreBoard").classList.toggle("cheat-on", e.target.checked);
});

document.getElementById("cardTextToggle").addEventListener("change", (e) => {
  document.getElementById("app").classList.toggle("show-card-text", e.target.checked);
});
