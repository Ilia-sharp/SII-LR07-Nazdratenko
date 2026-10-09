/* Логика сайта: статус сервера, прогресс обучения и оценки, /act, журнал.
   Только vanilla JS: fetch и setInterval. */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const STAGES = ["init", "train", "evaluate", "done"];
  const state = { busy: false, pollTimer: null, logTimer: null, trainEnabled: false, history: null, replayRaf: null };
  const REPLAY_MS = 16000; // длительность воспроизведения записи обучения

  // ---------- утилиты ----------

  function toast(text) {
    const box = $("toasts");
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = text;
    box.appendChild(el);
    setTimeout(() => { el.style.transition = "opacity 300ms"; el.style.opacity = "0"; }, 3200);
    setTimeout(() => el.remove(), 3600);
  }

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let data = null;
    try { data = await res.json(); } catch (_) { data = null; }
    if (!res.ok) {
      const msg = (data && data.detail) ? (typeof data.detail === "string" ? data.detail : "Ошибка запроса") : "Сервис вернул ошибку";
      const err = new Error(msg);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  function setStatus(kind, text) {
    const dot = $("statusDot");
    dot.className = "dot dot-" + kind;
    $("statusText").textContent = text;
  }

  function setBusy(on) {
    state.busy = on;
    ["btnTrain", "btnEval", "btnAct"].forEach((id) => {
      const b = $(id);
      if (b) b.disabled = on || (id === "btnTrain" && !state.trainEnabled && !state.history);
    });
  }

  function formatSeconds(s) {
    if (s == null) return "—";
    if (s < 60) return Math.round(s) + " с";
    return Math.floor(s / 60) + " мин " + Math.round(s % 60) + " с";
  }

  // ---------- статус сервера (холодный старт на Render Free) ----------

  async function waitForServer() {
    setStatus("wait", "Сервер просыпается…");
    $("trainBar") && null;
    let attempts = 0;
    while (true) {
      try {
        const h = await fetch("/health", { cache: "no-store" });
        if (h.ok) {
          const data = await h.json();
          $("modelVersion").textContent = data.model_version;
          setStatus("ok", "Онлайн");
          return data;
        }
      } catch (_) { /* сеть или холодный старт — повторим */ }
      attempts += 1;
      if (attempts > 40) {
        setStatus("err", "Недоступен");
        toast("Сервер недоступен. Попробуйте обновить страницу через минуту.");
        return null;
      }
      await new Promise((r) => setTimeout(r, 3000));
    }
  }

  async function loadStatusAndTrainFlag() {
    try {
      const st = await api("/api/status");
      state.trainEnabled = !!st.train_enabled;
    } catch (_) { /* не критично */ }

    if (state.trainEnabled) return;
    // Живое обучение на сервере отключено: показываем запись реального обучения
    const note = $("trainDisabledNote");
    note.classList.remove("hidden");
    try {
      state.history = await api("/api/training-history");
      const n = state.history.timesteps.toLocaleString("ru-RU");
      $("btnTrain").textContent = "Показать ход обучения";
      $("trainHint").textContent = "Воспроизведение записи реального обучения PPO в симуляторе (seed " + state.history.seed + ", " + n + " шагов).";
      note.textContent = "Бесплатный сервер не тянет обучение, поэтому здесь проигрывается запись настоящего обучения, сделанного заранее в симуляторе.";
      $("btnTrain").disabled = false;
    } catch (_) {
      note.textContent = "Обучение на этом сервере отключено, а записи обучения пока нет.";
      $("btnTrain").disabled = true;
    }
  }

  // ---------- прогресс ----------

  function renderProgress(p) {
    const percent = Math.round(p.percent || 0);
    $("trainBar").style.width = percent + "%";
    $("trainPercent").textContent = percent + "%";
    $("trainEta").textContent = "ETA " + formatSeconds(p.eta_seconds);
    $("liveReward").textContent = p.mean_reward != null ? p.mean_reward.toFixed(1) : "—";
    if (p.job_type === "train" && p.total) $("trainStepsText").textContent = "Шаги: " + (p.done || 0).toLocaleString("ru-RU") + " / " + p.total.toLocaleString("ru-RU");

    const idx = { init: 0, train: 1, evaluate: 2, done: 3, idle: -1, error: -1 }[p.stage];
    document.querySelectorAll("#trainSteps li").forEach((li, i) => {
      li.classList.remove("active", "done");
      if (idx >= 0) {
        if (i < idx) li.classList.add("done");
        if (i === idx) li.classList.add("active");
        if (p.stage === "done" && i === 3) li.classList.add("done");
      }
    });

    if (p.job_type === "evaluate" && p.total) {
      renderEvalSegments(p.done || 0, p.total / 2 || 20);
    }
  }

  function renderEvalSegments(doneSteps, totalEpisodes) {
    const box = $("evalSegments");
    if (box.children.length === 0) {
      for (let i = 0; i < 20; i++) box.appendChild(document.createElement("span"));
    }
    const perAgent = Math.min(doneSteps, 20);
    [...box.children].forEach((seg, i) => {
      seg.classList.toggle("filled", i < perAgent);
    });
  }

  function pollProgress(onDone) {
    clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      try {
        const p = await api("/api/progress");
        if (p.stage === "error") {
          clearInterval(state.pollTimer);
          toast(p.error || "Операция завершилась ошибкой");
          setBusy(false);
          return;
        }
        renderProgress(p);
        if (p.stage === "done" || p.stage === "idle") {
          clearInterval(state.pollTimer);
          setBusy(false);
          if (onDone) onDone(p);
        }
      } catch (err) {
        clearInterval(state.pollTimer);
        setBusy(false);
        toast(err.message || "Потеряна связь с сервером");
      }
    }, 1000);
  }

  // ---------- воспроизведение записи обучения ----------

  const CHART_W = 600, CHART_H = 170, CHART_TOP = 10, CHART_BOTTOM = 150, MAX_R = 500;

  function chartXY(step, total, reward) {
    const x = (step / total) * CHART_W;
    const y = CHART_BOTTOM - (Math.min(reward, MAX_R) / MAX_R) * (CHART_BOTTOM - CHART_TOP);
    return [x, y];
  }

  function drawChart(points, total) {
    if (!points.length) return;
    const xy = points.map((p) => chartXY(p.step, total, p.reward));
    $("chartLine").setAttribute("points", xy.map((q) => q[0].toFixed(1) + "," + q[1].toFixed(1)).join(" "));
    const last = xy[xy.length - 1];
    $("chartArea").setAttribute("d",
      "M" + xy[0][0].toFixed(1) + "," + CHART_BOTTOM + " L" +
      xy.map((q) => q[0].toFixed(1) + "," + q[1].toFixed(1)).join(" L") +
      " L" + last[0].toFixed(1) + "," + CHART_BOTTOM + " Z");
    $("chartDot").setAttribute("cx", last[0].toFixed(1));
    $("chartDot").setAttribute("cy", last[1].toFixed(1));
  }

  function setStageClasses(stage) {
    const idx = { init: 0, train: 1, evaluate: 2, done: 3 }[stage];
    document.querySelectorAll("#trainSteps li").forEach((li, i) => {
      li.classList.remove("active", "done");
      if (i < idx) li.classList.add("done");
      if (i === idx) li.classList.add(stage === "done" ? "done" : "active");
    });
  }

  function startReplay() {
    const h = state.history;
    if (!h || !h.points || !h.points.length) { toast("Записи обучения нет"); return; }
    setBusy(true);
    cancelAnimationFrame(state.replayRaf);

    // награда может быть null в начале: подставляем предыдущее значение (или 0)
    let carry = 0;
    const pts = h.points.map((p) => {
      if (p.mean_reward != null) carry = p.mean_reward;
      return { step: p.step, reward: carry };
    });
    const total = h.timesteps;
    $("chartLine").setAttribute("points", "");
    $("chartArea").setAttribute("d", "");
    $("liveReward").textContent = "—";

    const t0 = performance.now();
    function frame(now) {
      const t = Math.min(1, (now - t0) / REPLAY_MS);
      // фазы: инициализация 0–4%, обучение 4–90%, оценка 90–97%, готово
      let stage = "init", trainFrac = 0;
      if (t >= 0.97) { stage = "done"; trainFrac = 1; }
      else if (t >= 0.90) { stage = "evaluate"; trainFrac = 1; }
      else if (t >= 0.04) { stage = "train"; trainFrac = (t - 0.04) / 0.86; }

      const curStep = Math.round(trainFrac * total);
      const shown = pts.filter((p) => p.step <= curStep);
      drawChart(shown, total);
      const lastR = shown.length ? shown[shown.length - 1].reward : 0;

      $("trainBar").style.width = (t * 100).toFixed(1) + "%";
      $("trainPercent").textContent = Math.round(t * 100) + "%";
      $("trainStepsText").textContent = "Шаги: " + curStep.toLocaleString("ru-RU") + " / " + total.toLocaleString("ru-RU");
      $("trainEta").textContent = "Осталось ~" + Math.max(0, Math.ceil((1 - t) * REPLAY_MS / 1000)) + " с";
      $("liveReward").textContent = shown.length ? lastR.toFixed(1) : "—";
      setStageClasses(stage);

      if (t < 1) {
        state.replayRaf = requestAnimationFrame(frame);
      } else {
        if (h.final_eval_mean != null) $("liveReward").textContent = h.final_eval_mean.toFixed(1);
        $("trainEta").textContent = "Готово";
        setBusy(false);
        toast("Обучение завершено: награда " + (h.final_eval_mean != null ? h.final_eval_mean.toFixed(0) : lastR.toFixed(0)) + " из 500");
      }
    }
    state.replayRaf = requestAnimationFrame(frame);
  }

  // ---------- обучение ----------

  async function startTrain() {
    if (state.busy) return;
    if (!state.trainEnabled) { startReplay(); return; }
    setBusy(true);
    $("trainBar").style.width = "0%";
    try {
      await api("/api/train?timesteps=30000", { method: "POST" });
      pollProgress(() => toast("Обучение-кандидат завершено"));
    } catch (err) {
      setBusy(false);
      toast(err.message);
    }
  }

  // ---------- оценка ----------

  async function startEval() {
    if (state.busy) return;
    setBusy(true);
    $("compare").classList.add("hidden");
    $("ratio").classList.add("hidden");
    renderEvalSegments(0, 20);
    try {
      await api("/api/evaluate", { method: "POST" });
      pollProgress(async () => {
        try {
          const r = await api("/api/evaluate/result");
          showCompare(r);
        } catch (err) {
          toast(err.message);
        }
      });
    } catch (err) {
      setBusy(false);
      toast(err.message);
    }
  }

  function showCompare(r) {
    $("compare").classList.remove("hidden");
    $("ppoMean").textContent = r.ppo.mean.toFixed(1);
    $("rndMean").textContent = r.random.mean.toFixed(1);
    const max = Math.max(r.ppo.mean, r.random.mean, 1);
    requestAnimationFrame(() => {
      $("ppoBar").style.width = (100 * r.ppo.mean / max) + "%";
      $("rndBar").style.width = (100 * r.random.mean / max) + "%";
    });
    if (r.improvement_ratio) {
      $("ratio").textContent = "PPO лучше случайного в " + r.improvement_ratio + " раз";
      $("ratio").classList.remove("hidden");
    }
  }

  // ---------- /act ----------

  async function callAct() {
    const state4 = [0, 1, 2, 3].map((i) => Number($("s" + i).value));
    if (state4.some((x) => !Number.isFinite(x))) {
      toast("Введите четыре числа");
      return;
    }
    const body = { state: state4 };
    $("btnAct").disabled = true;
    try {
      const res = await api("/act", { method: "POST", body: JSON.stringify(body) });
      const big = $("actBig");
      big.textContent = res.action === 0 ? "Действие: 0 (влево)" : "Действие: 1 (вправо)";
      big.className = "result-big " + (res.action === 0 ? "act-left" : "act-right");
      $("actVersion").textContent = "model_version: " + res.model_version;
      $("actJson").textContent =
        "POST /act\n" + JSON.stringify(body, null, 2) + "\n\nОтвет:\n" + JSON.stringify(res, null, 2);
      loadLog();
    } catch (err) {
      toast(err.message);
    } finally {
      $("btnAct").disabled = false;
    }
  }

  // ---------- журнал ----------

  function fmtState(arr) {
    return "[" + arr.map((x) => Number(x).toFixed(2)).join(", ") + "]";
  }

  async function loadLog() {
    try {
      const data = await api("/api/logs?limit=10");
      const body = $("logBody");
      if (!data.items.length) {
        body.innerHTML = '<tr><td colspan="4" class="muted">Записей пока нет. Выполните запрос /act.</td></tr>';
        return;
      }
      body.innerHTML = "";
      data.items.forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + (r.time || "").replace("T", " ").replace("Z", "") + "</td>" +
          '<td class="state">' + fmtState(r.state || []) + "</td>" +
          "<td>" + r.action + "</td>" +
          "<td>" + (r.reward == null ? "—" : r.reward) + "</td>";
        body.appendChild(tr);
      });
    } catch (_) {
      /* тихо: журнал обновится в следующий раз */
    }
  }

  // ---------- старт ----------

  document.addEventListener("DOMContentLoaded", async () => {
    $("btnTrain").addEventListener("click", startTrain);
    $("btnEval").addEventListener("click", startEval);
    $("btnAct").addEventListener("click", callAct);

    const h = await waitForServer();
    if (h) {
      await loadStatusAndTrainFlag();
      loadLog();
      state.logTimer = setInterval(loadLog, 3000);
      // если задача уже идёт (например, после обновления страницы) — подхватываем прогресс
      try {
        const p = await api("/api/progress");
        if (p.stage !== "idle" && p.stage !== "done" && p.stage !== "error") {
          setBusy(true);
          pollProgress(null);
        }
      } catch (_) { /* ignore */ }
    }
  });
})();
