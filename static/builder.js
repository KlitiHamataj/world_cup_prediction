// Custom bracket: the user fills the Round of 32, then the model resolves each
// tie one match at a time (with a short suspense) and the winner advances toward
// the final in the centre — same symmetric layout as the Knockout bracket page.
(function () {
  function start() {
    let FLAGS = {};
    try { FLAGS = JSON.parse(document.getElementById("flags-data").textContent || "{}"); } catch (e) {}

    const ROUND_COUNT = { 0: 8, 1: 4, 2: 2, 3: 1 }; // matches per half, per round
    const bracket = document.querySelector(".bracket");
    const playBtn = document.getElementById("play-btn");
    const resetBtn = document.getElementById("reset-btn");
    const statusEl = document.getElementById("builder-status");
    const champion = document.getElementById("champion");
    if (!bracket || !playBtn) return;

    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const id = (half, ri, idx) => half + "-" + ri + "-" + idx;
    const matchEl = (half, ri, idx) => document.getElementById(id(half, ri, idx));
    const teamEl = (el, slot) => el.querySelector('.team[data-slot="' + slot + '"]');
    const withFlag = (name) => (FLAGS[name] ? FLAGS[name] + " " + name : name);

    // ---- play order: round by round, both halves -------------------------
    const PLAN = [];
    for (let ri = 0; ri <= 3; ri++) {
      for (const half of ["L", "R"]) {
        for (let idx = 0; idx < ROUND_COUNT[ri]; idx++) PLAN.push({ half: half, ri: ri, idx: idx });
      }
    }
    PLAN.push({ half: "F", ri: 4, idx: 0 }); // the final

    function getTeam(half, ri, idx, slot) {
      const el = teamEl(matchEl(half, ri, idx), slot);
      if (ri === 0) return el.querySelector(".bk-select").value;
      return el.dataset.team || "";
    }

    function setTeam(el, slot, name) {
      const t = teamEl(el, slot);
      t.dataset.team = name;
      t.querySelector(".nm").innerHTML = withFlag(name);
    }

    function advance(winner, half, ri, idx) {
      if (ri < 3) {
        const parent = matchEl(half, ri + 1, Math.floor(idx / 2));
        parent.classList.remove("pending");
        setTeam(parent, idx % 2, winner);
      } else if (ri === 3) {
        const fin = matchEl("F", 4, 0);
        fin.classList.remove("pending");
        setTeam(fin, half === "L" ? 0 : 1, winner);
      } else {
        champion.classList.add("revealed");
        champion.querySelector(".champ-name").innerHTML = withFlag(winner);
      }
    }

    // ---- validation ------------------------------------------------------
    function validate() {
      const picks = [];
      bracket.querySelectorAll('.ko-match[data-round="0"] .bk-select').forEach((s) => picks.push(s.value));
      if (picks.some((p) => !p)) return "Fill in all 32 teams first.";
      const counts = {};
      picks.forEach((p) => (counts[p] = (counts[p] || 0) + 1));
      const dup = Object.keys(counts).filter((t) => counts[t] > 1);
      if (dup.length) return "Each team can appear only once. Duplicate: " + dup.join(", ");
      return null;
    }

    // ---- play one tie ----------------------------------------------------
    async function playMatch(half, ri, idx) {
      const el = matchEl(half, ri, idx);
      const t1 = getTeam(half, ri, idx, 0);
      const t2 = getTeam(half, ri, idx, 1);

      el.classList.add("playing");
      try { el.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" }); } catch (e) {}
      statusEl.textContent = t1 + " vs " + t2 + "…";
      await sleep(750);

      let data;
      try {
        const res = await fetch("/api/match?team1=" + encodeURIComponent(t1) + "&team2=" + encodeURIComponent(t2));
        data = await res.json();
        if (data.error) throw new Error(data.error);
      } catch (e) {
        el.classList.remove("playing");
        statusEl.textContent = "Error: " + e.message;
        statusEl.classList.add("err");
        throw e;
      }

      const s0 = teamEl(el, 0), s1 = teamEl(el, 1);
      const pc0 = s0.querySelector(".pc"), pc1 = s1.querySelector(".pc");
      if (pc0) pc0.textContent = Math.round(data.p_home * 100) + "%";
      if (pc1) pc1.textContent = Math.round(data.p_away * 100) + "%";
      const winFirst = data.winner === t1;
      (winFirst ? s0 : s1).classList.add("win");
      (winFirst ? s1 : s0).classList.add("lose");
      if (data.coin_flip) el.classList.add("coin");
      el.classList.remove("playing");
      el.classList.add("done");

      advance(data.winner, half, ri, idx);
      drawLines();
      if (window.renderFlags) window.renderFlags();
      await sleep(450);
    }

    async function play() {
      const err = validate();
      if (err) { statusEl.textContent = err; statusEl.classList.add("err"); return; }
      statusEl.classList.remove("err");
      playBtn.disabled = true;
      bracket.classList.add("locked");
      bracket.querySelectorAll(".bk-select").forEach((s) => (s.disabled = true));

      for (const m of PLAN) {
        await playMatch(m.half, m.ri, m.idx);
      }

      statusEl.textContent = "🏆 Champion: " + champion.querySelector(".champ-name").textContent.trim();
      resetBtn.hidden = false;
      playBtn.hidden = true;
    }

    // ---- connector lines (same as the bracket page) ----------------------
    function pos(el, wrap, wrapRect) {
      const r = el.getBoundingClientRect();
      const left = r.left - wrapRect.left + wrap.scrollLeft;
      const top = r.top - wrapRect.top + wrap.scrollTop;
      return { left: left, right: left + r.width, top: top, bottom: top + r.height,
               cx: left + r.width / 2, cy: top + r.height / 2 };
    }
    function line(svg, d) {
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", d);
      svg.appendChild(p);
    }
    function drawLines() {
      const wrap = document.querySelector(".bracket-wrap");
      const svg = document.querySelector(".connectors");
      if (!wrap || !bracket || !svg) return;
      const wrapRect = wrap.getBoundingClientRect();
      const W = bracket.scrollWidth, H = bracket.scrollHeight;
      svg.setAttribute("width", W); svg.setAttribute("height", H);
      svg.style.width = W + "px"; svg.style.height = H + "px";
      svg.innerHTML = "";
      const final = matchEl("F", 4, 0);
      document.querySelectorAll(".ko-match[data-half]").forEach((m) => {
        const half = m.dataset.half, ri = +m.dataset.round, idx = +m.dataset.idx;
        if (half === "F") {
          if (!champion) return;
          const a = pos(m, wrap, wrapRect), b = pos(champion, wrap, wrapRect);
          line(svg, "M " + a.cx + " " + a.top + " V " + b.bottom);
          return;
        }
        const parent = (ri === 3) ? final : matchEl(half, ri + 1, Math.floor(idx / 2));
        if (!parent) return;
        const a = pos(m, wrap, wrapRect), b = pos(parent, wrap, wrapRect);
        let x1, x2;
        if (half === "L") { x1 = a.right; x2 = b.left; } else { x1 = a.left; x2 = b.right; }
        const midx = (x1 + x2) / 2;
        line(svg, "M " + x1 + " " + a.cy + " H " + midx + " V " + b.cy + " H " + x2);
      });
    }

    // ---- randomize: drop 32 distinct teams into the Round of 32 ----------
    function randomize() {
      const selects = bracket.querySelectorAll('.ko-match[data-round="0"] .bk-select');
      if (!selects.length) return;
      const pool = Array.from(selects[0].options).map((o) => o.value).filter(Boolean);
      for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
      selects.forEach((s, k) => { s.value = pool[k % pool.length]; });
      statusEl.textContent = "Teams randomized — hit Play.";
      statusEl.classList.remove("err");
    }

    playBtn.addEventListener("click", play);
    const randomBtn = document.getElementById("random-btn");
    if (randomBtn) randomBtn.addEventListener("click", randomize);
    resetBtn.addEventListener("click", () => window.location.reload());
    window.addEventListener("load", drawLines);
    window.addEventListener("resize", function () {
      clearTimeout(window.__bkt);
      window.__bkt = setTimeout(drawLines, 120);
    });
    drawLines();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
