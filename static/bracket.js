// Draw connector lines for the symmetric bracket (left + right halves -> centre final).
(function () {
  function pos(el, wrap, wrapRect) {
    const r = el.getBoundingClientRect();
    const left = r.left - wrapRect.left + wrap.scrollLeft;
    const top = r.top - wrapRect.top + wrap.scrollTop;
    return {
      left: left, right: left + r.width,
      top: top, bottom: top + r.height,
      cx: left + r.width / 2, cy: top + r.height / 2,
    };
  }

  function line(svg, d) {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    svg.appendChild(p);
  }

  function draw() {
    const wrap = document.querySelector(".bracket-wrap");
    const bracket = document.querySelector(".bracket");
    const svg = document.querySelector(".connectors");
    if (!wrap || !bracket || !svg) return;

    const wrapRect = wrap.getBoundingClientRect();
    const W = bracket.scrollWidth, H = bracket.scrollHeight;
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    svg.style.width = W + "px";
    svg.style.height = H + "px";
    svg.innerHTML = "";

    const matches = Array.from(document.querySelectorAll(".ko-match[data-half]"));
    const final = document.getElementById("F-4-0");
    const champion = document.getElementById("champion");

    matches.forEach((m) => {
      const half = m.dataset.half;
      const ri = +m.dataset.round, idx = +m.dataset.idx;

      if (half === "F") {
        if (!champion) return;
        const a = pos(m, wrap, wrapRect), b = pos(champion, wrap, wrapRect);
        line(svg, "M " + a.cx + " " + a.top + " V " + b.bottom);
        return;
      }

      const parent = (ri === 3)
        ? final
        : document.getElementById(half + "-" + (ri + 1) + "-" + Math.floor(idx / 2));
      if (!parent) return;

      const a = pos(m, wrap, wrapRect), b = pos(parent, wrap, wrapRect);
      let x1, x2;
      if (half === "L") { x1 = a.right; x2 = b.left; }
      else { x1 = a.left; x2 = b.right; }
      const midx = (x1 + x2) / 2;
      line(svg, "M " + x1 + " " + a.cy + " H " + midx + " V " + b.cy + " H " + x2);
    });
  }

  window.addEventListener("load", draw);
  window.addEventListener("resize", function () {
    clearTimeout(window.__bt);
    window.__bt = setTimeout(draw, 120);
  });
})();
