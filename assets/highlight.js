/* ============================================================
   highlight.js — 零依赖语法高亮（file:// 直接可用）
   支持 python / bash / cpp / text，自动识别。
   只处理 <pre><code> 里的纯文本节点，保留作者手写的
   <span class="cm">（中文注释）与 <span class="file">（文件名）。
   ============================================================ */

(function () {
  "use strict";

  /* ---------- 词法规则 ---------- */

  // 1 注释  2 字符串  3 装饰器  4 关键字  5 数字  6 调用  7 self/super
  var PY = /(#.*$)|([frbu]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(@[\w.]+)|(\b(?:def|class|return|if|else|elif|for|while|in|not|and|or|import|from|as|with|try|except|finally|raise|pass|break|continue|lambda|yield|assert|global|nonlocal|async|await|del|is|None|True|False)\b)|(\b\d[\d_]*(?:\.\d+)?\b)|(\b[A-Za-z_]\w*(?=\())|(\b(?:self|super)\b)/gm;

  // 1 注释  2 字符串  3 变量  4 关键字  5 参数  6 数字
  var SH = /(#.*$)|('(?:[^']*)'|"(?:\\.|[^"\\])*")|(\$\{?\w+\}?)|(\b(?:if|then|else|elif|fi|for|while|do|done|case|esac|function|export|local|return|cd|echo|set|source)\b)|(\s--?[A-Za-z][\w-]*)|(\b\d+\b)/gm;

  // 1 行注释  2 块注释  3 字符串  4 关键字  5 数字  6 预处理
  var CPP = /(\/\/.*$)|(\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\])*")|(\b(?:class|struct|template|typename|public|private|protected|void|int|float|double|bool|char|const|auto|return|if|else|for|while|namespace|using|new|delete|nullptr|true|false|static|inline|virtual|override|constexpr)\b)|(\b\d+\b)|(#\s*\w+)/gm;

  var RULES = { py: PY, bash: SH, cpp: CPP };

  /* ---------- 语言识别 ---------- */

  function detect(text) {
    var t = text.replace(/\r/g, "");

    // shell
    if (/^\s*(\$|nvidia-smi|pip |cd |export |git |python |hf |huggingface-cli|wsl)/m.test(t)) return "bash";
    if (/DEEPINFER_[A-Z_]+\s*=/.test(t) && !/\bdef\s/.test(t)) return "bash";
    if (/(^|\n)\s*(--?[A-Za-z][\w-]*\s+)+\S+/m.test(t) && !/[:=]/.test(t)) return "bash";

    // C/C++
    if (/(#include|std::|template\s*<|\bnamespace\b|::\w+\()/.test(t)) return "cpp";

    // Python
    if (/(^|\n)\s*(def |class |import |from [\w.]+ import|@torch|@triton)/.test(t)) return "py";
    if (/\bself\.\w+/.test(t)) return "py";
    if (/^\s*\w+\s*:\s*(list|dict|tuple|set|str|int|float|bool|Optional|Callable|tuple\[|\[)/m.test(t)) return "py";
    if (/\b(for \w+ in|if .+ else|return |yield |lambda )/.test(t)) return "py";
    if (/\w+\s*=\s*[\w.]+\(.*\)/.test(t)) return "py";
    // 函数参数列表：name=value, 每行一条
    if (/^\s{2,}\w+\s*=\s*[^=,\n]+,\s*$/m.test(t) && !/^\s*[A-Z_]{3,}\s*=/m.test(t)) return "py";

    return "text";
  }

  /* ---------- 渲染 ---------- */

  function span(cls, text) {
    var el = document.createElement("span");
    el.className = cls;
    el.textContent = text;
    return el;
  }

  function fold(text, lang) {
    var frag = document.createDocumentFragment();
    var re = RULES[lang];
    if (!re) {
      frag.appendChild(document.createTextNode(text));
      return frag;
    }
    re.lastIndex = 0;
    var last = 0;
    var m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) {
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      }
      var cls = null;
      if (m[1] != null) cls = m[0].charAt(0) === "#" ? "tok-cm" : "tok-cm";
      else if (m[2] != null) cls = "tok-str";
      else if (m[3] != null) cls = lang === "py" ? "tok-dec" : "tok-var";
      else if (m[4] != null) cls = "tok-kw";
      else if (m[5] != null) cls = lang === "bash" ? "tok-flag" : "tok-num";
      else if (m[6] != null) cls = lang === "py" ? "tok-fn" : "tok-num";
      else if (m[7] != null) cls = "tok-bi";
      if (lang === "cpp" && m[6] != null) cls = "tok-pp";

      frag.appendChild(cls ? span(cls, m[0]) : document.createTextNode(m[0]));

      // def / class 后面的名字单独上色
      if (cls === "tok-kw" && /^(def|class)$/.test(m[0])) {
        var rest = text.slice(re.lastIndex);
        var nm = /^(\s+)([A-Za-z_]\w*)/.exec(rest);
        if (nm) {
          frag.appendChild(document.createTextNode(nm[1]));
          frag.appendChild(span("tok-fn", nm[2]));
          re.lastIndex += nm[0].length;
        }
      }

      last = re.lastIndex;
      if (m[0] === "") re.lastIndex++; // 防御：空匹配
    }
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    return frag;
  }

  function highlight(pre) {
    var code = pre.querySelector("code");
    if (!code || code.dataset.highlighted === "1") return;

    // 语言识别时排除文件名标签，避免 "xxx.py · L1–def f()" 破坏行首匹配
    var probe = code.cloneNode(true);
    Array.prototype.forEach.call(probe.querySelectorAll(".file"), function (n) {
      n.parentNode.removeChild(n);
    });
    var lang = detect(probe.textContent);
    code.dataset.lang = lang;

    // 收集可被处理的文本节点：跳过 .cm / .file 子树
    var walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        var p = node.parentElement;
        while (p && p !== code) {
          if (p.classList.contains("cm") || p.classList.contains("file")) {
            return NodeFilter.FILTER_REJECT;
          }
          p = p.parentElement;
        }
        return node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });

    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    nodes.forEach(function (node) {
      node.parentNode.replaceChild(fold(node.nodeValue, lang), node);
    });

    code.dataset.highlighted = "1";
  }

  function highlightAll() {
    var blocks = document.querySelectorAll("pre");
    for (var i = 0; i < blocks.length; i++) highlight(blocks[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", highlightAll);
  } else {
    highlightAll();
  }

  window.SyntaxHighlight = { highlightAll: highlightAll };
})();
