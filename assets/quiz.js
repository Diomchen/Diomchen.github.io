/* ============================================================
   quiz.js — 共享测验组件（无依赖，file:// 可直接用）
   用法：
     <div class="quiz" data-quiz="engine-loop"></div>
     <script src="../assets/quiz.js"></script>
     <script>
       CourseQuiz.mount('[data-quiz="engine-loop"]', {
         questions: [{ q, options: [...], answer: 0, why: "..." }]
       });
     </script>
   ============================================================ */

(function () {
  "use strict";

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  // 把 `code` 转成 <code>，其余文本转义。题目和解析里都用反引号包代码。
  function fmt(text) {
    var escaped = String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    return escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function mount(target, config) {
    var root = typeof target === "string" ? document.querySelector(target) : target;
    if (!root) return;

    var questions = config.questions || [];
    var answered = 0;
    var correct = 0;
    var scoreLine = el("p", "score", "");

    function updateScore() {
      var tail;
      if (answered < questions.length) {
        tail = "已答 " + answered + " / " + questions.length;
      } else if (correct === questions.length) {
        tail = "全部正确。现在合上页面，把结论复述一遍。";
      } else if (correct >= questions.length - 1) {
        tail = "接近了。回去看错的那一题对应的源码行。";
      } else {
        tail = "回到对应小节重读一遍，然后重来。";
      }
      scoreLine.textContent = "得分 " + correct + " / " + questions.length + " · " + tail;
    }

    questions.forEach(function (item, index) {
      var block = el("div", "quiz-q");
      var heading = el("p", "q-text");
      heading.innerHTML = index + 1 + ". " + fmt(item.q);
      block.appendChild(heading);

      var options = el("div", "options");
      var feedback = el("div", "feedback");
      feedback.setAttribute("aria-live", "polite");

      item.options.forEach(function (label, optionIndex) {
        var button = el("button", "option");
        button.type = "button";
        button.appendChild(el("span", "tag", "ABCD"[optionIndex] + "."));
        var labelSpan = el("span", "label-text");
        labelSpan.innerHTML = fmt(label);
        button.appendChild(labelSpan);

        button.addEventListener("click", function () {
          if (block.dataset.done === "1") return;
          block.dataset.done = "1";
          answered += 1;

          var isRight = optionIndex === item.answer;
          if (isRight) correct += 1;

          Array.prototype.forEach.call(options.children, function (child, i) {
            child.disabled = true;
            if (i === item.answer) child.classList.add("correct");
            if (i === optionIndex && !isRight) child.classList.add("wrong");
          });

          feedback.className = "feedback " + (isRight ? "ok" : "no");
          feedback.innerHTML = (isRight ? "对。" : "再看一眼。") + fmt(item.why || "");
          updateScore();
        });

        options.appendChild(button);
      });

      block.appendChild(options);
      block.appendChild(feedback);
      root.appendChild(block);
    });

    root.appendChild(scoreLine);
    updateScore();
  }

  window.CourseQuiz = { mount: mount };
})();
