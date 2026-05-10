const STORAGE_KEY = "vivascope.session.v1";

const els = {
  healthDot: document.querySelector("#healthDot"),
  healthText: document.querySelector("#healthText"),
  setupForm: document.querySelector("#setupForm"),
  sampleBtn: document.querySelector("#sampleBtn"),
  clearInputsBtn: document.querySelector("#clearInputsBtn"),
  resetBtn: document.querySelector("#resetBtn"),
  setupSummary: document.querySelector("#setupSummary"),
  startBtn: document.querySelector("#startBtn"),
  lengthShortLabel: document.querySelector("#lengthShortLabel"),
  lengthStandardLabel: document.querySelector("#lengthStandardLabel"),
  lengthDeepLabel: document.querySelector("#lengthDeepLabel"),
  lengthHint: document.querySelector("#lengthHint"),
  scenario: document.querySelector("#scenario"),
  style: document.querySelector("#style"),
  major: document.querySelector("#major"),
  targetProfile: document.querySelector("#targetProfile"),
  project: document.querySelector("#project"),
  focus: document.querySelector("#focus"),
  emptyState: document.querySelector("#emptyState"),
  sessionView: document.querySelector("#sessionView"),
  modeLabel: document.querySelector("#modeLabel"),
  styleLabel: document.querySelector("#styleLabel"),
  roundLabel: document.querySelector("#roundLabel"),
  phaseBadge: document.querySelector("#phaseBadge"),
  warningBox: document.querySelector("#warningBox"),
  loadingOverlay: document.querySelector("#loadingOverlay"),
  loadingTitle: document.querySelector("#loadingTitle"),
  loadingHint: document.querySelector("#loadingHint"),
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
  activeInsight: "risk",
};

const modeLabels = {
  project: "只练项目经历抗追问能力",
  mixed: "综合模拟",
  knowledge: "只练基础知识回答能力",
};

const baseLengthLabels = {
  short: "快速",
  standard: "标准",
  deep: "深度",
};

const baseRoundCounts = {
  short: 3,
  standard: 6,
  deep: 9,
};

const phaseLabels = {
  project: "项目追问",
  knowledge: "基础知识问诊",
};

const startBtnDefaultText = "生成训练";
const answerBtnDefaultText = "提交回答";

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

function setBusy(isBusy, label = "处理中", hint = "") {
  state.busy = isBusy;
  document.body.classList.toggle("is-busy", isBusy);
  setControlsDisabled(isBusy);
  if (isBusy) {
    els.startBtn.querySelector("span").textContent = label;
    els.answerBtn.querySelector("span").textContent = label;
    els.loadingOverlay.classList.remove("hidden");
    els.loadingTitle.textContent = label;
    els.loadingHint.textContent = hint || "面试官正在分析材料和上一轮回答，请稍等。";
  } else {
    els.startBtn.querySelector("span").textContent = startBtnDefaultText;
    els.answerBtn.querySelector("span").textContent = answerBtnDefaultText;
    els.loadingOverlay.classList.add("hidden");
  }
}

function setControlsDisabled(disabled) {
  [
    ...els.setupForm.querySelectorAll("input, select, textarea, button"),
    ...els.answerForm.querySelectorAll("textarea, button"),
    els.sampleBtn,
    els.clearInputsBtn,
    els.resetBtn,
  ].forEach((control) => {
    if (control) control.disabled = disabled;
  });
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
  const form = new FormData(els.setupForm);
  return {
    mode: form.get("mode"),
    interview_length: form.get("interview_length") || "standard",
    scenario: els.scenario.value,
    style: els.style.value,
    major: els.major.value.trim(),
    target_profile: els.targetProfile.value.trim(),
    project: els.project.value.trim(),
    focus: els.focus.value.trim(),
  };
}

function roundsForMode(mode, length) {
  const base = baseRoundCounts[length || "standard"] || baseRoundCounts.standard;
  return mode === "mixed" ? base * 2 : base;
}

function lengthLabelForMode(mode, length) {
  const key = length || "standard";
  return `${baseLengthLabels[key] || "标准"} ${roundsForMode(mode, key)} 轮`;
}

function lengthLabelForSession(session) {
  const key = session.input.interview_length || "standard";
  return `${baseLengthLabels[key] || "标准"} ${session.max_rounds} 轮`;
}

function updateLengthLabels() {
  const mode = getSelectedMode();
  els.lengthShortLabel.textContent = lengthLabelForMode(mode, "short");
  els.lengthStandardLabel.textContent = lengthLabelForMode(mode, "standard");
  els.lengthDeepLabel.textContent = lengthLabelForMode(mode, "deep");
  els.lengthHint.textContent =
    mode === "mixed"
      ? "综合模拟 = 完整项目追问 + 完整基础问诊：6、12、18 轮。"
      : "单项训练可选快速、标准、深度：3、6、9 轮。";
}

function updateSegmentedIndicators() {
  document.querySelectorAll(".segmented-control").forEach((control) => {
    const inputs = [...control.querySelectorAll('input[type="radio"]')];
    const index = Math.max(0, inputs.findIndex((input) => input.checked));
    control.dataset.activeIndex = String(index);
  });
}

function updateCharCount(field) {
  if (!field) return;
  const counter = document.querySelector(`[data-counter-for="${field.id}"]`);
  if (!counter) return;
  const max = field.getAttribute("maxlength") || "∞";
  counter.textContent = `${field.value.length} / ${max}`;
  counter.classList.toggle("near-limit", field.maxLength > 0 && field.value.length > field.maxLength * 0.86);
}

function updateCharCounts() {
  [els.major, els.targetProfile, els.project, els.focus, els.answerInput].forEach(updateCharCount);
}

function initCharCounters() {
  [els.major, els.targetProfile, els.project, els.focus, els.answerInput].forEach((field) => {
    field?.addEventListener("input", () => updateCharCount(field));
  });
  updateCharCounts();
}

function fillFormFromSession(session) {
  if (!session) return;
  const input = session.input;
  const modeInput = document.querySelector(`input[name="mode"][value="${input.mode}"]`);
  if (modeInput) modeInput.checked = true;
  const lengthInput = document.querySelector(`input[name="interview_length"][value="${input.interview_length || "standard"}"]`);
  if (lengthInput) lengthInput.checked = true;
  els.scenario.value = input.scenario;
  els.style.value = input.style;
  els.major.value = input.major;
  els.targetProfile.value = input.target_profile || "";
  els.project.value = input.project || "";
  els.focus.value = input.focus || "";
  updateLengthLabels();
  updateSegmentedIndicators();
  updateCharCounts();
}

function renderIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function render() {
  const session = state.session;
  document.body.classList.toggle("has-session", Boolean(session));
  document.body.classList.remove("editing-setup");
  if (!session) {
    els.emptyState.classList.remove("hidden");
    els.sessionView.classList.add("hidden");
    els.setupSummary.classList.add("hidden");
    els.setupForm.classList.remove("hidden");
    renderIcons();
    return;
  }

  els.emptyState.classList.add("hidden");
  els.sessionView.classList.remove("hidden");
  renderSetupSummary(session);
  els.modeLabel.textContent = modeLabels[session.input.mode] || "训练";
  els.styleLabel.textContent = `${session.input.scenario} · ${session.input.style}`;
  const nextRound = Math.min(session.turns.length + 1, session.max_rounds);
  const phase = getRoundPhase(session, nextRound);
  els.roundLabel.textContent =
    session.status === "finished" ? `已完成 ${session.max_rounds} 轮` : `第 ${nextRound} / ${session.max_rounds} 轮`;
  els.phaseBadge.textContent = session.status === "finished" ? "复盘完成" : phaseLabels[phase];

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

function getRoundPhase(session, roundNumber) {
  const mode = session.input.mode;
  if (mode === "project") return "project";
  if (mode === "knowledge") return "knowledge";
  const projectRounds = Math.max(1, Math.floor(Number(session.max_rounds || 6) / 2));
  return Number(roundNumber) <= projectRounds ? "project" : "knowledge";
}

function describePhasePlan(session) {
  if (session.input.mode === "project") return `全程 ${session.max_rounds} 轮项目追问`;
  if (session.input.mode === "knowledge") return `全程 ${session.max_rounds} 轮基础知识问诊`;
  const projectRounds = Math.max(1, Math.floor(Number(session.max_rounds || roundsForMode("mixed", session.input.interview_length)) / 2));
  return `${projectRounds} 轮项目追问 + ${session.max_rounds - projectRounds} 轮基础知识问诊`;
}

function renderSetupSummary(session) {
  const input = session.input;
  const topRisk = [...(session.risk_radar || [])].sort((a, b) => b.level - a.level)[0];
  const projectTitle = session.project_map?.theme || input.project || "基础知识问诊";
  const shortProjectTitle = projectTitle.length > 150 ? `${projectTitle.slice(0, 150)}...` : projectTitle;
  els.setupSummary.classList.remove("hidden");
  els.setupForm.classList.add("hidden");
  els.setupSummary.innerHTML = `
    <div class="summary-card">
      <p class="eyebrow">本轮设置</p>
      <h3>${escapeHtml(modeLabels[input.mode] || "训练")}</h3>
      <div class="summary-tags">
        <span>${escapeHtml(input.scenario)}</span>
        <span>${escapeHtml(input.style)}</span>
        <span>${escapeHtml(lengthLabelForSession(session))}</span>
      </div>
      <dl>
        <div>
          <dt>专业背景</dt>
          <dd>${escapeHtml(input.major)}</dd>
        </div>
        ${
          input.target_profile
            ? `<div><dt>申请目标</dt><dd>${escapeHtml(input.target_profile)}</dd></div>`
            : ""
        }
        <div>
          <dt>${input.mode === "knowledge" ? "训练重点" : "项目摘要"}</dt>
          <dd>${escapeHtml(shortProjectTitle)}</dd>
        </div>
        ${
          topRisk
            ? `<div><dt>最高风险</dt><dd>${escapeHtml(topRisk.dimension)} · ${topRisk.level}/5</dd></div>`
            : ""
        }
        <div>
          <dt>阶段安排</dt>
          <dd>${escapeHtml(describePhasePlan(session))}</dd>
        </div>
      </dl>
      <div class="summary-actions">
        <button id="summaryEditBtn" class="secondary-btn" type="button">
          <i data-lucide="sliders-horizontal"></i>
          <span>展开设置</span>
        </button>
        <button id="summaryResetBtn" class="ghost-btn" type="button">
          <i data-lucide="rotate-ccw"></i>
          <span>重新开始</span>
        </button>
      </div>
    </div>
  `;
}

function renderInsights(session) {
  const tabs = getInsightTabs(session);
  if (!tabs.length) {
    els.insightGrid.innerHTML = "";
    return;
  }
  if (!tabs.some((tab) => tab.id === state.activeInsight)) {
    state.activeInsight = tabs[0].id;
  }
  const active = tabs.find((tab) => tab.id === state.activeInsight) || tabs[0];
  const summary = buildInsightSummary(session);

  els.insightGrid.innerHTML = `
    <div class="insight-summary">
      <p class="eyebrow">追问依据</p>
      <h3>${escapeHtml(summary.title)}</h3>
      <p>${escapeHtml(summary.text)}</p>
    </div>
    <div class="insight-tabs" role="tablist">
      ${tabs
        .map(
          (tab) => `
            <button class="${tab.id === active.id ? "active" : ""}" type="button" data-insight-tab="${tab.id}">
              ${escapeHtml(tab.label)}
            </button>
          `,
        )
        .join("")}
    </div>
    <article class="insight-section">${active.render()}</article>
  `;
}

function getInsightTabs(session) {
  const tabs = [];
  if (session.risk_radar?.length) {
    tabs.push({
      id: "risk",
      label: "风险",
      render: () => `
        <h3>追问风险摘要</h3>
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
      `,
    });
  }

  if (session.project_map) {
    const map = session.project_map;
    tabs.push({
      id: "map",
      label: "脉络",
      render: () => `
        <h3>项目脉络图</h3>
        <ul class="map-list">
          <li><strong>项目主题</strong>${escapeHtml(map.theme)}</li>
          <li><strong>项目动机</strong>${escapeHtml(map.motivation)}</li>
          <li><strong>主要方法</strong>${escapeHtml((map.methods || []).join("；") || "待补充")}</li>
          <li><strong>实验/实现依据</strong>${escapeHtml((map.evidence || []).join("；") || "待补充")}</li>
          <li><strong>结果与贡献</strong>${escapeHtml([...(map.results || []), ...(map.contribution || [])].join("；") || "待补充")}</li>
        </ul>
      `,
    });
  }

  if (session.knowledge_points?.length) {
    tabs.push({
      id: "knowledge",
      label: "知识",
      render: () => `
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
      `,
    });
  }
  return tabs;
}

function buildInsightSummary(session) {
  const topRisk = [...(session.risk_radar || [])].sort((a, b) => b.level - a.level)[0];
  if (topRisk) {
    return {
      title: `${topRisk.dimension} · ${topRisk.level}/5`,
      text: topRisk.reason,
    };
  }
  const point = session.knowledge_points?.[0];
  if (point) {
    return {
      title: point.name,
      text: point.why_relevant,
    };
  }
  return {
    title: "等待训练分析",
    text: "生成训练后，这里会显示本轮追问的主要依据。",
  };
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
  const turns = [...session.turns].reverse();
  els.historyPanel.innerHTML = `
    <div class="history-heading">
      <div>
        <p class="eyebrow">Review Timeline</p>
        <h3>回答复盘时间线</h3>
      </div>
      <span class="score-pill">${session.turns.length} / ${session.max_rounds} 轮</span>
    </div>
    <div class="turn-timeline">
      ${turns
        .map((turn, index) => {
          const phase = getRoundPhase(session, turn.round_index);
          return `
            <details class="turn-detail" ${index === 0 ? "open" : ""}>
              <summary>
                <span>第 ${turn.round_index} 轮 · ${escapeHtml(phaseLabels[phase])}</span>
                <p>${escapeHtml(turn.question)}</p>
                <strong>${turn.feedback.score} 分</strong>
              </summary>
              <div class="turn-card">
                <div class="qa-block compact">
                  <p><strong>问：</strong>${escapeHtml(turn.question)}</p>
                  <p><strong>答：</strong>${escapeHtml(turn.answer)}</p>
                </div>
                <div class="feedback-grid">
                  ${feedbackBox("亮点", turn.feedback.strengths)}
                  ${feedbackBox("漏洞", turn.feedback.gaps)}
                  ${feedbackBox("补充点", turn.feedback.suggestions)}
                  ${feedbackBox("框架", turn.feedback.answer_frame)}
                </div>
                ${renderScoreExplain(turn.feedback)}
              </div>
            </details>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderScoreExplain(feedback) {
  if (!feedback.score_reason && !feedback.rewrite) return "";
  return `
    <div class="score-explain">
      ${
        feedback.score_reason
          ? `<div><h4>评分理由</h4><p>${escapeHtml(feedback.score_reason)}</p></div>`
          : ""
      }
      ${
        feedback.rewrite
          ? `<div><h4>参考改写</h4><p>${escapeHtml(feedback.rewrite)}</p></div>`
          : ""
      }
    </div>
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
    <div class="report-hero">
      <div>
        <p class="eyebrow">Final Review</p>
        <h3>最终复盘报告</h3>
      </div>
      <p class="report-score"><strong>${report.total_score}</strong><span>/ 100</span></p>
    </div>
    <div class="report-summary">
      <div>
        <h4>回答概览</h4>
        <p>${escapeHtml(report.answer_summary || report.closing_comment)}</p>
      </div>
      <div>
        <h4>总体建议</h4>
        <p>${escapeHtml(report.overall_advice || "下一轮优先补齐证据链、个人贡献边界和项目局限。")}</p>
      </div>
    </div>
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
  setBusy(true, "生成训练中", "正在提取项目脉络、标记追问风险，并准备第一轮问题。");
  try {
    const response = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.session = response.session;
    state.activeInsight = "risk";
    saveSession(state.session);
    els.answerInput.value = "";
    updateCharCount(els.answerInput);
    render();
  } catch (error) {
    alert(error.message);
  } finally {
    setBusy(false);
  }
});

els.setupForm.addEventListener("change", (event) => {
  if (state.busy) return;
  if (event.target.name === "mode") {
    updateLengthLabels();
  }
  if (event.target.type === "radio") {
    updateSegmentedIndicators();
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
  const isFinalRound = state.session.turns.length + 1 >= state.session.max_rounds;
  setBusy(
    true,
    isFinalRound ? "生成复盘中" : "分析回答中",
    isFinalRound ? "正在整理所有回答并生成整体报告。" : "面试官正在分析你的回答，并准备下一轮追问。",
  );
  try {
    const response = await api(`/api/sessions/${state.session.session_id}/answer`, {
      method: "POST",
      body: JSON.stringify({ answer }),
    });
    state.session = response.session;
    saveSession(state.session);
    els.answerInput.value = "";
    updateCharCount(els.answerInput);
    render();
  } catch (error) {
    alert(error.message);
  } finally {
    setBusy(false);
  }
});

els.sampleBtn.addEventListener("click", () => {
  if (state.busy) return;
  document.querySelector('input[name="mode"][value="mixed"]').checked = true;
  document.querySelector('input[name="interview_length"][value="standard"]').checked = true;
  els.scenario.value = "保研复试";
  els.style.value = "严格导师型";
  els.major.value = "人工智能专业，大三，做过机器学习和计算机视觉课程项目";
  els.targetProfile.value =
    "保研申请智能感知与机器人实验室，导师方向包括计算机视觉、多模态感知和机器人操作，希望重点展示自己具备可靠实验设计、模型理解和工程落地能力。";
  els.project.value =
    "我参与了一个基于深度学习的实验室安全帽佩戴检测项目，目标是在实验室监控画面中识别人员是否正确佩戴安全帽。项目使用公开数据集和我们补充采集的少量实验室图片，先做数据清洗和标注，然后用 YOLO 系列目标检测模型训练。我的主要工作是整理数据、完成训练脚本、调整数据增强参数，并对比不同输入尺寸和置信度阈值下的检测效果。最终在测试集上 mAP 有一定提升，但在光照较暗和遮挡场景下仍然误检较多。我担心老师会追问为什么选择 YOLO、数据量不大是否可靠、以及这个项目的创新点到底是什么。";
  els.focus.value = "重点训练方法选择、实验可靠性、结果不好怎么解释、个人贡献和创新点不足。";
  updateLengthLabels();
  updateSegmentedIndicators();
  updateCharCounts();
});

els.clearInputsBtn.addEventListener("click", () => {
  if (state.busy) return;
  els.major.value = "";
  els.targetProfile.value = "";
  els.project.value = "";
  els.focus.value = "";
  updateCharCounts();
  els.major.focus();
});

async function resetSession() {
  if (state.busy) return;
  const sessionId = state.session?.session_id;
  state.session = null;
  state.activeInsight = "risk";
  clearSavedSession();
  els.answerInput.value = "";
  updateCharCount(els.answerInput);
  render();
  if (sessionId) {
    try {
      await api(`/api/sessions/${sessionId}`, { method: "DELETE" });
    } catch {
      // Local reset should still succeed when the old backend session is already gone.
    }
  }
}

els.resetBtn.addEventListener("click", resetSession);

els.setupSummary.addEventListener("click", (event) => {
  if (state.busy) return;
  const target = event.target.closest("button");
  if (!target) return;
  if (target.id === "summaryResetBtn") {
    resetSession();
  }
  if (target.id === "summaryEditBtn") {
    document.body.classList.add("editing-setup");
    els.setupSummary.classList.add("hidden");
    els.setupForm.classList.remove("hidden");
  }
});

els.insightGrid.addEventListener("click", (event) => {
  const tab = event.target.closest("[data-insight-tab]");
  if (!tab) return;
  state.activeInsight = tab.dataset.insightTab;
  renderInsights(state.session);
  renderIcons();
});

checkHealth();
restoreSavedSession().then(render);
updateLengthLabels();
updateSegmentedIndicators();
initCharCounters();
renderIcons();
