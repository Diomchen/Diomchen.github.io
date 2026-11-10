/* ============================================================
   lesson-ui.js — 阅读体验增强（零依赖）
   1. 代码块「复制」按钮
   2. 自动生成本课目录（h2）
   3. 顶部阅读进度条
   ============================================================ */

(function () {
  "use strict";

  /* ---------- 复制按钮 ---------- */

  function codeText(pre) {
    var code = pre.querySelector("code");
    if (!code) return "";
    var clone = code.cloneNode(true);
    var label = clone.querySelector(".file");
    if (label) label.remove();
    return clone.textContent.replace(/\s+$/, "");
  }

  function legacyCopy(text, done) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta);
    done();
  }

  function addCopyButtons() {
    var pres = document.querySelectorAll("pre");
    for (var i = 0; i < pres.length; i++) {
      (function (pre) {
        if (pre.querySelector(".copy-btn")) return;
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "copy-btn no-print";
        btn.setAttribute("aria-label", "复制这段代码");
        btn.textContent = "复制";

        btn.addEventListener("click", function () {
          var text = codeText(pre);
          var settled = false;
          var finish = function () {
            if (settled) return;
            settled = true;
            btn.textContent = "已复制";
            btn.classList.add("copied");
            window.setTimeout(function () {
              btn.textContent = "复制";
              btn.classList.remove("copied");
            }, 1400);
          };
          try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
              navigator.clipboard.writeText(text).then(finish, function () {
                legacyCopy(text, finish);
              });
              // 某些环境里 clipboard 的 promise 既不 resolve 也不 reject，兜底
              window.setTimeout(function () {
                if (!settled) legacyCopy(text, finish);
              }, 250);
            } else {
              legacyCopy(text, finish);
            }
          } catch (e) {
            legacyCopy(text, finish);
          }
        });

        pre.appendChild(btn);
      })(pres[i]);
    }
  }

  /* ---------- 本课目录 ---------- */

  function addToc() {
    var page = document.querySelector(".page");
    if (!page || document.querySelector(".lesson-toc")) return;

    var heads = Array.prototype.filter.call(page.querySelectorAll("h2"), function (h) {
      return h.closest(".page") === page && h.id !== "toc";
    });
    if (heads.length < 3) return;

    var nav = document.createElement("nav");
    nav.className = "lesson-toc no-print";
    nav.setAttribute("aria-label", "本课目录");

    var label = document.createElement("span");
    label.className = "label";
    label.textContent = "本课目录";
    nav.appendChild(label);

    heads.forEach(function (h, i) {
      if (!h.id) h.id = "s-" + (i + 1);
      var clone = h.cloneNode(true);
      var num = clone.querySelector(".num");
      if (num) num.remove();
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = clone.textContent.trim();
      nav.appendChild(a);
    });

    var meta = page.querySelector(".meta");
    if (meta && meta.parentNode) {
      meta.parentNode.insertBefore(nav, meta.nextSibling);
    } else {
      page.insertBefore(nav, page.firstChild);
    }
  }

  /* ---------- 阅读进度条 ---------- */

  function addProgress() {
    if (document.querySelector(".reading-progress")) return;
    var bar = document.createElement("div");
    bar.className = "reading-progress no-print";
    bar.setAttribute("aria-hidden", "true");
    document.body.appendChild(bar);

    var ticking = false;
    function update() {
      var doc = document.documentElement;
      var max = doc.scrollHeight - window.innerHeight;
      var pct = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      bar.style.width = (pct * 100).toFixed(2) + "%";
      ticking = false;
    }
    window.addEventListener("scroll", function () {
      if (!ticking) {
        ticking = true;
        window.requestAnimationFrame(update);
      }
    }, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  function init() {
    addCopyButtons();
    addToc();
    addProgress();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
