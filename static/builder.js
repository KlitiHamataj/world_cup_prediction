// Custom bracket: the user fills the Round of 32, then the model resolves each
// tie one match at a time (with a short suspense) and the winner advances toward
// the final in the centre — same symmetric layout as the Knockout bracket page.
(function () {
  function start() {
    let FLAGS = {};
    try { FLAGS = JSON.parse(document.getElementById("flags-data").textContent || "{}"); } catch (e) {}
    let TEAMS = [];
    try { TEAMS = JSON.parse(document.getElementById("teams-data").textContent || "[]"); } catch (e) {}

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
    // FLAGS maps team -> ISO code (e.g. "de"); render the same <img> flag the
    // server macro uses so advancing / randomized / swapped teams match the
    // initial Round-of-32 cards instead of showing the bare country code.
    const withFlag = (name) => {
      const code = FLAGS[name];
      if (!code) return name;
      return '<img class="flag" src="https://flagcdn.com/' + code + '.svg" ' +
             'alt="' + name + '" loading="lazy"> ' + name;
    };

    // ---- play order: round by round, both halves -------------------------
    const PLAN = [];
    for (let ri = 0; ri <= 3; ri++) {
      for (const half of ["L", "R"]) {
        for (let idx = 0; idx < ROUND_COUNT[ri]; idx++) PLAN.push({ half: half, ri: ri, idx: idx });
      }
    }
    PLAN.push({ half: "F", ri: 4, idx: 0 }); // the final

    function getTeam(half, ri, idx, slot) {
      return teamEl(matchEl(half, ri, idx), slot).dataset.team || "";
    }

    function setCardTeam(card, name) {
      card.dataset.team = name;
      card.querySelector(".nm").innerHTML = withFlag(name);
    }

    function setTeam(el, slot, name) {
      setCardTeam(teamEl(el, slot), name);
    }

    const benchCards = () => Array.from(document.querySelectorAll("#bench .bench-card"));
    const allDraggable = () => Array.from(document.querySelectorAll(".team.draggable"));

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
    function r32Cards() {
      return Array.from(bracket.querySelectorAll('.ko-match[data-round="0"] .team'));
    }

    function validate() {
      const picks = r32Cards().map((c) => c.dataset.team || "");
      if (picks.some((p) => !p)) return "Every slot needs a team.";
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
      document.querySelector(".builder-page").classList.add("locked");
      allDraggable().forEach((c) => { c.setAttribute("draggable", "false"); c.classList.remove("draggable"); });

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
      const cards = r32Cards(), bench = benchCards();
      if (!cards.length) return;
      const pool = (TEAMS.length ? TEAMS.slice() : cards.concat(bench).map((c) => c.dataset.team));
      for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
      cards.forEach((c, k) => setCardTeam(c, pool[k]));            // first 32 -> bracket
      bench.forEach((c, k) => setCardTeam(c, pool[cards.length + k])); // rest -> bench
      if (window.renderFlags) window.renderFlags();
      statusEl.textContent = "Teams randomized — drag to tweak, then Play.";
      statusEl.classList.remove("err");
    }

    // ---- drag & drop: swap the two dragged team cards --------------------
    let dragSrc = null;
    function clearDragOver() {
      bracket.querySelectorAll(".team.drag-over").forEach((x) => x.classList.remove("drag-over"));
    }
    function onDragStart(e) {
      const card = e.currentTarget;
      if (!card.classList.contains("draggable")) { e.preventDefault(); return; }
      dragSrc = card;
      card.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
      try { e.dataTransfer.setData("text/plain", card.dataset.team || ""); } catch (_) {}
    }
    function onDragOver(e) {
      if (!dragSrc) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      if (e.currentTarget !== dragSrc) e.currentTarget.classList.add("drag-over");
    }
    function onDragLeave(e) { e.currentTarget.classList.remove("drag-over"); }
    function onDrop(e) {
      e.preventDefault();
      const target = e.currentTarget;
      target.classList.remove("drag-over");
      if (!dragSrc || dragSrc === target) return;
      const a = dragSrc.dataset.team, b = target.dataset.team;
      setCardTeam(dragSrc, b);
      setCardTeam(target, a);
      if (window.renderFlags) window.renderFlags();
    }
    function onDragEnd(e) {
      e.currentTarget.classList.remove("dragging");
      clearDragOver();
      dragSrc = null;
    }
    allDraggable().forEach((card) => {
      card.addEventListener("dragstart", onDragStart);
      card.addEventListener("dragover", onDragOver);
      card.addEventListener("dragleave", onDragLeave);
      card.addEventListener("drop", onDrop);
      card.addEventListener("dragend", onDragEnd);
    });

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
