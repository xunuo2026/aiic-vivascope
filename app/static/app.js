const STORAGE_KEY = "vivascope.session.v1";

const els = {
  healthDot: document.querySelector("#healthDot"),
  healthText: document.querySelector("#healthText"),
  setupForm: document.querySelector("#setupForm"),
  sampleBtn: document.querySelector("#sampleBtn"),
  resetBtn: document.querySelector("#resetBtn"),
  startBtn: document.querySelector("#startBtn"),
  scenario: document.querySelector("#scenario"),
  style: document.querySelector("#style"),
  major: document.querySelector("#major"),
  project: document.querySelector("#project"),
  focus: document.querySelector("#focus"),
  emptyState: document.querySelector("#emptyState"),
  sessionView: document.querySelector("#sessionView"),
  modeLabel: document.querySelector("#modeLabel"),
  styleLabel: document.querySelector("#styleLabel"),
  roundLabel: document.querySelector("#roundLabel"),
  warningBox: document.querySelector("#warningBox"),
  insightGrid: document.querySelector("#insightGrid"),
  currentQuestion: document.querySelector("#currentQuestion"),
  answerForm: document.querySelector("#answerForm"),
  answerInput: document.querySelector("#answerInput"),
  answerBtn: document.querySelector("#answerBtn"),
  questionPanel: document.querySelector("#questionPanel"),
  historyPanel: document.querySelector("#historyPanel"),
  reportPanel: document.querySelector("#reportPanel"),
};

const state = {
  session: null,
  busy: false,
};

const modeLabels = {
  project: "只练项目经历抗追问能力",
  mixed: "综合模拟",
  knowledge: "只练基础知识回答能力",
};

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function listHtml(items = []) {
  return items.length ? items.map((item) => `<li>${escapeHtml(item)}</li>`).join("") : "<li>暂无</li>";
}

function setBusy(isBusy, label = "处理中") {
  state.busy = isBusy;
  els.startBtn.disabled = isBusy;
  els.answerBtn.disabled = isBusy;
  if (isBusy) {
    els.startBtn.querySelector("span").textContent = label;
    els.answerBtn.querySelector("span").textContent = label;
  } else {
    els.startBtn.querySelector("span").textContent = "生成训练";
    els.answerBtn.querySelector("span").textContent = "提交回答";
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    const message = payload.detail || "请求失败，请稍后重试。";
    throw new Error(Array.isArray(message) ? message.map((item) => item.msg).join("；") : message);
  }
  return payload;
}

function saveSession(session) {
  if (!session) return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

function clearSavedSession() {
  localStorage.removeItem(STORAGE_KEY);
}

function getSelectedMode() {
  return new FormData(els.setupForm).get("mode");
}

function collectStartPayload() {
  return {
    mode: getSelectedMode(),
    scenario: els.scenario.value,
    style: els.style.value,
    major: els.major.value.trim(),
    project: els.project.value.trim(),
    focus: els.focus.value.trim(),
  };
}

function fillFormFromSession(session) {
  if (!session) return;
  const input = session.input;
  const modeInput = document.querySelector(`input[name="mode"][value="${input.mode}"]`);
  if (modeInput) modeInput.checked = true;
  els.scenario.value = input.scenario;
  els.style.value = input.style;
  els.major.value = input.major;
  els.project.value = input.project || "";
  els.focus.value = input.focus || "";
}

function renderIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function render() {
  const session = state.session;
  if (!session) {
    els.emptyState.classList.remove("hidden");
    els.sessionView.classList.add("hidden");
    renderIcons();
    return;
  }

  els.emptyState.classList.add("hidden");
  els.sessionView.classList.remove("hidden");
  els.modeLabel.textContent = modeLabels[session.input.mode] || "训练";
  els.styleLabel.textContent = `${session.input.scenario} · ${session.input.style}`;
  const nextRound = Math.min(session.turns.length + 1, session.max_rounds);
  els.roundLabel.textContent =
    session.status === "finished" ? `已完成 ${session.max_rounds} 轮` : `第 ${nextRound} / ${session.max_rounds} 轮`;

  if (session.ai_warning) {
    els.warningBox.classList.remove("hidden");
    els.warningBox.textContent = session.ai_warning;
  } else {
    els.warningBox.classList.add("hidden");
  }

  renderInsights(session);
  renderQuestion(session);
  renderHistory(session);
  renderReport(session);
  renderIcons();
}

function renderInsights(session) {
  const sections = [];

  if (session.project_map) {
    const map = session.project_map;
    sections.push(`
      <article class="insight-section">
        <h3>项目脉络图</h3>
        <ul class="map-list">
          <li><strong>项目主题</strong>${escapeHtml(map.theme)}</li>
          <li><strong>项目动机</strong>${escapeHtml(map.motivation)}</li>
          <li><strong>主要方法</strong>${escapeHtml((map.methods || []).join("；") || "待补充")}</li>
          <li><strong>实验/实现依据</strong>${escapeHtml((map.evidence || []).join("；") || "待补充")}</li>
          <li><strong>结果与贡献</strong>${escapeHtml([...(map.results || []), ...(map.contribution || [])].join("；") || "待补充")}</li>
        </ul>
      </article>
    `);
  }

  if (session.risk_radar?.length) {
    sections.push(`
      <article class="insight-section">
        <h3>追问风险雷达</h3>
        <div class="risk-list">
          ${session.risk_radar
            .map((risk) => {
              const width = Math.max(20, Math.min(100, Number(risk.level) * 20));
              const high = risk.level >= 4 ? " high" : "";
              return `
                <div class="risk-item${high}">
                  <div class="risk-title">
                    <span>${escapeHtml(risk.dimension)}</span>
                    <span>${risk.level}/5</span>
                  </div>
                  <div class="risk-track"><div class="risk-bar" style="width:${width}%"></div></div>
                  <p>${escapeHtml(risk.reason)}</p>
                </div>
              `;
            })
            .join("")}
        </div>
      </article>
    `);
  }

  if (session.knowledge_points?.length) {
    sections.push(`
      <article class="insight-section">
        <h3>知识点清单</h3>
        <ul class="knowledge-list">
          ${session.knowledge_points
            .map(
              (point) => `
                <li>
                  <strong>${escapeHtml(point.name)}</strong>
                  ${escapeHtml(point.why_relevant)}
                  <br />
                  <span>${escapeHtml(point.probe_example)}</span>
                </li>
              `,
            )
            .join("")}
        </ul>
      </article>
    `);
  }

  els.insightGrid.innerHTML = sections.join("");
}

function renderQuestion(session) {
  if (session.status === "finished") {
    els.questionPanel.classList.add("hidden");
    return;
  }
  els.questionPanel.classList.remove("hidden");
  els.currentQuestion.textContent = session.current_question;
}

function renderHistory(session) {
  if (!session.turns.length) {
    els.historyPanel.innerHTML = "";
    return;
  }
  els.historyPanel.innerHTML = `
    <h3>即时反馈</h3>
    ${[...session.turns]
      .reverse()
      .map(
        (turn) => `
          <article class="turn-card">
            <div class="turn-head">
              <strong>第 ${turn.round_index} 轮</strong>
              <span class="score-pill">${turn.feedback.score} 分</span>
            </div>
            <div class="qa-block">
              <p><strong>问：</strong>${escapeHtml(turn.question)}</p>
              <p><strong>答：</strong>${escapeHtml(turn.answer)}</p>
            </div>
            <div class="feedback-grid">
              ${feedbackBox("亮点", turn.feedback.strengths)}
              ${feedbackBox("漏洞", turn.feedback.gaps)}
              ${feedbackBox("建议补充", turn.feedback.suggestions)}
              ${feedbackBox("回答框架", turn.feedback.answer_frame)}
            </div>
          </article>
        `,
      )
      .join("")}
  `;
}

function feedbackBox(title, items) {
  return `
    <div class="feedback-box">
      <h4>${escapeHtml(title)}</h4>
      <ul>${listHtml(items)}</ul>
    </div>
  `;
}

function renderReport(session) {
  if (session.status !== "finished" || !session.final_report) {
    els.reportPanel.classList.add("hidden");
    els.reportPanel.innerHTML = "";
    return;
  }
  const report = session.final_report;
  els.reportPanel.classList.remove("hidden");
  els.reportPanel.innerHTML = `
    <h3>最终复盘报告</h3>
    <p class="report-score"><strong>${report.total_score}</strong><span>/ 100</span></p>
    <div class="report-grid">
      ${reportBlock("最容易被问穿的点", report.most_vulnerable_project_points)}
      ${reportBlock("知识薄弱点", report.knowledge_weaknesses)}
      ${reportBlock("表达问题", report.expression_issues)}
      ${reportBlock("下一轮训练任务", report.next_training_tasks)}
    </div>
    <div class="report-block">
      <h4>复盘结论</h4>
      <p>${escapeHtml(report.closing_comment)}</p>
    </div>
  `;
}

function reportBlock(title, items) {
  return `
    <div class="report-block">
      <h4>${escapeHtml(title)}</h4>
      <ul class="report-list">${listHtml(items)}</ul>
    </div>
  `;
}

async function checkHealth() {
  try {
    const health = await api("/health");
    els.healthDot.className = health.api_key_configured && health.model_configured ? "status-dot ok" : "status-dot warn";
    els.healthText.textContent =
      health.api_key_configured && health.model_configured ? "千问已配置" : "演示模式";
  } catch {
    els.healthDot.className = "status-dot error";
    els.healthText.textContent = "后端未连接";
  }
}

async function restoreSavedSession() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const saved = JSON.parse(raw);
    const response = await api(`/api/sessions/${saved.session_id}`);
    state.session = response.session;
  } catch {
    try {
      const saved = JSON.parse(raw);
      const response = await api("/api/sessions/restore", {
        method: "POST",
        body: JSON.stringify({ session: saved }),
      });
      state.session = response.session;
    } catch {
      clearSavedSession();
    }
  }
  fillFormFromSession(state.session);
  render();
}

els.setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const payload = collectStartPayload();
  if (!payload.major) {
    alert("请先填写专业背景。");
    return;
  }
  if ((payload.mode === "project" || payload.mode === "mixed") && payload.project.length < 30) {
    alert("项目追问或综合模拟需要至少填写一段项目经历。");
    return;
  }
  setBusy(true, "生成中");
  try {
    const response = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.session = response.session;
    saveSession(state.session);
    els.answerInput.value = "";
    render();
  } catch (error) {
    alert(error.message);
  } finally {
    setBusy(false);
  }
});

els.answerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy || !state.session) return;
  const answer = els.answerInput.value.trim();
  if (!answer) {
    alert("请先输入本轮回答。");
    return;
  }
  setBusy(true, "追问中");
  try {
    const response = await api(`/api/sessions/${state.session.session_id}/answer`, {
      method: "POST",
      body: JSON.stringify({ answer }),
    });
    state.session = response.session;
    saveSession(state.session);
    els.answerInput.value = "";
    render();
  } catch (error) {
    alert(error.message);
  } finally {
    setBusy(false);
  }
});

els.sampleBtn.addEventListener("click", () => {
  document.querySelector('input[name="mode"][value="mixed"]').checked = true;
  els.scenario.value = "保研复试";
  els.style.value = "严格导师型";
  els.major.value = "人工智能专业，大三，做过机器学习和计算机视觉课程项目";
  els.project.value =
    "我参与了一个基于深度学习的实验室安全帽佩戴检测项目，目标是在实验室监控画面中识别人员是否正确佩戴安全帽。项目使用公开数据集和我们补充采集的少量实验室图片，先做数据清洗和标注，然后用 YOLO 系列目标检测模型训练。我的主要工作是整理数据、完成训练脚本、调整数据增强参数，并对比不同输入尺寸和置信度阈值下的检测效果。最终在测试集上 mAP 有一定提升，但在光照较暗和遮挡场景下仍然误检较多。我担心老师会追问为什么选择 YOLO、数据量不大是否可靠、以及这个项目的创新点到底是什么。";
  els.focus.value = "重点训练方法选择、实验可靠性、结果不好怎么解释、个人贡献和创新点不足。";
});

els.resetBtn.addEventListener("click", async () => {
  const sessionId = state.session?.session_id;
  state.session = null;
  clearSavedSession();
  els.answerInput.value = "";
  render();
  if (sessionId) {
    try {
      await api(`/api/sessions/${sessionId}`, { method: "DELETE" });
    } catch {
      // Local reset should still succeed when the old backend session is already gone.
    }
  }
});

checkHealth();
restoreSavedSession().then(render);
renderIcons();

