(function () {
  const isCurrentSeasonWindow = /_(?:l3|l5|l10)(?:_|\.html)/i.test(window.location.pathname);
  const state = {
    activeYearIndex: 0,
    droppedDimension: null,
    originalData: null,
    playerSearch: "",
  };
  const PLAYER_SEARCH_TRACE_NAME = "Player search highlight";

  function getGraph() {
    return document.querySelector(".plotly-graph-div");
  }

  function readArray(value) {
    if (!value) return [];
    if (typeof value === "object" && typeof value.bdata === "string" && typeof value.dtype === "string") {
      const binary = window.atob(value.bdata);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }

      const constructors = {
        f8: Float64Array,
        f4: Float32Array,
        i1: Int8Array,
        i2: Int16Array,
        i4: Int32Array,
        u1: Uint8Array,
        u2: Uint16Array,
        u4: Uint32Array,
      };
      const TypedArray = constructors[value.dtype];
      return TypedArray ? Array.from(new TypedArray(bytes.buffer)) : [];
    }
    if (ArrayBuffer.isView(value)) return Array.from(value);
    return Array.isArray(value) ? value.slice() : value;
  }

  function getSceneTitle(gd, axis, fallback) {
    const title = gd && gd.layout && gd.layout.scene && gd.layout.scene[axis] && gd.layout.scene[axis].title;
    if (typeof title === "string") return title;
    return title && title.text ? title.text : fallback;
  }

  function getDimensions(gd) {
    return [
      { key: "pc1", axis: "x", sceneAxis: "xaxis", label: getSceneTitle(gd, "xaxis", "Dimension 1") },
      { key: "pc2", axis: "y", sceneAxis: "yaxis", label: getSceneTitle(gd, "yaxis", "Dimension 2") },
      { key: "pc3", axis: "z", sceneAxis: "zaxis", label: getSceneTitle(gd, "zaxis", "Dimension 3") },
    ];
  }

  function ensureOriginalData(gd) {
    if (state.originalData) return true;
    if (!gd || !gd.data || !gd.data.length) return false;

    const firstTrace = gd.data[0];
    if (!firstTrace || !firstTrace.x || !firstTrace.y || !firstTrace.z) return false;

    state.originalData = gd.data.map((trace) => ({
      x: readArray(trace.x),
      y: readArray(trace.y),
      z: readArray(trace.z),
      hovertext: readArray(trace.hovertext),
      markerSize: trace.marker && trace.marker.size ? readArray(trace.marker.size) : null,
      markerLineColor: trace.marker && trace.marker.line ? readArray(trace.marker.line.color) : null,
      markerLineWidth: trace.marker && trace.marker.line ? readArray(trace.marker.line.width) : null,
    }));

    return true;
  }

  function playerNameFromHover(value) {
    return String(value || "").replace(/\s+\(\d{4}\)\s*$/, "").trim();
  }

  function getBaseTraces(gd) {
    return state.originalData ? gd.data.slice(0, state.originalData.length) : gd.data;
  }

  function getPlayerSearchMatches(gd, query) {
    const normalizedQuery = query.toLowerCase().trim();
    if (!normalizedQuery || !state.originalData) return [];

    const matches = [];
    getBaseTraces(gd).forEach((trace, traceIndex) => {
      const source = state.originalData[traceIndex];
      if (!source) return;

      source.hovertext.forEach((label, pointIndex) => {
        const playerName = playerNameFromHover(label);
        const searchable = `${playerName} ${label}`.toLowerCase();
        if (!searchable.includes(normalizedQuery)) return;
        matches.push({ traceIndex, pointIndex, playerName, label });
      });
    });
    return matches;
  }

  function updatePlayerSearchStatus(matchCount) {
    const status = document.getElementById("player-search-status");
    if (!status) return;

    if (!state.playerSearch.trim()) {
      status.textContent = "";
      return;
    }

    status.textContent = matchCount === 1 ? "1 match" : `${matchCount} matches`;
  }

  function highlightedValue(values, pointIndex) {
    const value = values[pointIndex];
    return typeof value === "number" && Number.isFinite(value) ? value : value;
  }

  function buildPlayerSearchTrace(matches) {
    const dimensions = getDimensions(getGraph());
    const keptDimensions = dimensions.filter((dimension) => dimension.key !== state.droppedDimension);
    const isProjection = Boolean(state.droppedDimension);
    const trace = {
      name: PLAYER_SEARCH_TRACE_NAME,
      type: isProjection ? "scatter" : "scatter3d",
      mode: "markers",
      hoverinfo: "text",
      hovertext: matches.map((match) => match.label),
      text: [],
      textposition: isProjection ? "top center" : "top center",
      textfont: {
        color: "#f8fafc",
        size: isProjection ? 13 : 12,
        family: "Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
      },
      showlegend: false,
      marker: {
        color: "rgba(183, 255, 0, 0)",
        size: isProjection ? 10 : 8,
        opacity: 1,
        line: { color: "#b7ff00", width: 2 },
      },
    };

    trace.x = matches.map((match) => {
      const source = state.originalData[match.traceIndex];
      return highlightedValue(source[isProjection ? keptDimensions[0].axis : "x"], match.pointIndex);
    });
    trace.y = matches.map((match) => {
      const source = state.originalData[match.traceIndex];
      return highlightedValue(source[isProjection ? keptDimensions[1].axis : "y"], match.pointIndex);
    });
    if (!isProjection) {
      trace.z = matches.map((match) => highlightedValue(state.originalData[match.traceIndex].z, match.pointIndex));
    }

    return trace;
  }

  function buildPlayerSearchAnnotations() {
    return [];
  }

  function getPlayerSearchLayout(gd, matches) {
    const layout = {
      ...gd.layout,
      scene: { ...(gd.layout.scene || {}) },
    };
    const annotations = matches.length ? buildPlayerSearchAnnotations() : [];

    if (state.droppedDimension) {
      layout.annotations = annotations;
      layout.scene.annotations = [];
    } else {
      layout.annotations = [];
      layout.scene.annotations = annotations;
    }

    return layout;
  }

  function applyPlayerSearchHighlight() {
    const gd = getGraph();
    if (!gd || !window.Plotly || !ensureOriginalData(gd)) return;

    const query = state.playerSearch.toLowerCase().trim();
    const matches = query ? getPlayerSearchMatches(gd, query) : [];
    updatePlayerSearchStatus(matches.length);

    const data = getBaseTraces(gd).map((trace) => ({ ...trace }));
    if (query && matches.length > 0) {
      data.push(buildPlayerSearchTrace(matches));
    }

    Plotly.react(gd, data, getPlayerSearchLayout(gd, matches), {
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
    });
  }

  function getProjectionTrace(trace, index, keptDimensions) {
    const nextTrace = { ...trace };
    const source = state.originalData[index];

    delete nextTrace.scene;
    delete nextTrace.z;
    nextTrace.type = "scatter";
    nextTrace.mode = trace.mode || "markers";
    nextTrace.x = source[keptDimensions[0].axis];
    nextTrace.y = source[keptDimensions[1].axis];
    nextTrace.marker = { ...(trace.marker || {}) };

    return nextTrace;
  }

  function getRestoredTrace(trace, index) {
    const nextTrace = { ...trace };
    const source = state.originalData[index];

    delete nextTrace.xaxis;
    delete nextTrace.yaxis;
    nextTrace.type = "scatter3d";
    nextTrace.mode = trace.mode || "markers";
    nextTrace.x = source.x;
    nextTrace.y = source.y;
    nextTrace.z = source.z;
    nextTrace.marker = { ...(trace.marker || {}) };

    return nextTrace;
  }

  function getProjectedAxis(label, titleStandoff) {
    return {
      title: { text: label, font: { color: "#f8fafc", size: 15 }, standoff: titleStandoff },
      tickfont: { color: "#f8fafc", size: 12 },
      automargin: true,
      showline: false,
      mirror: false,
      zeroline: true,
      zerolinecolor: "rgba(229, 231, 235, 0.82)",
      zerolinewidth: 2,
      showgrid: true,
      gridcolor: "rgba(229, 231, 235, 0.28)",
      gridwidth: 1,
      ticks: "",
    };
  }

  function applyProjection() {
    const gd = getGraph();
    if (!gd || !window.Plotly) return;

    if (!ensureOriginalData(gd)) return;

    const dimensions = getDimensions(gd);
    const keptDimensions = dimensions.filter((dimension) => dimension.key !== state.droppedDimension);
    const data = getBaseTraces(gd).map((trace, index) => (
      state.droppedDimension
        ? getProjectionTrace(trace, index, keptDimensions)
        : getRestoredTrace(trace, index)
    ));
    const baseMargin = gd.layout.margin || {};
    const layout = {
      ...gd.layout,
      margin: state.droppedDimension
        ? {
          ...baseMargin,
          t: Math.max(baseMargin.t || 0, 154),
          r: Math.max(baseMargin.r || 0, 64),
          b: Math.max(baseMargin.b || 0, 340),
        }
        : { ...baseMargin },
      legend: state.droppedDimension
        ? {
          ...(gd.layout.legend || {}),
          orientation: "h",
          x: 0.5,
          xanchor: "center",
          y: -0.64,
          yanchor: "top",
        }
        : { ...(gd.layout.legend || {}) },
      scene: {
        ...(gd.layout.scene || {}),
        xaxis: { ...((gd.layout.scene || {}).xaxis || {}), title: { text: dimensions[0].label }, showspikes: false },
        yaxis: { ...((gd.layout.scene || {}).yaxis || {}), title: { text: dimensions[1].label }, showspikes: false },
        zaxis: { ...((gd.layout.scene || {}).zaxis || {}), title: { text: dimensions[2].label }, showspikes: false },
        dragmode: "turntable",
      },
      xaxis: state.droppedDimension
        ? getProjectedAxis(keptDimensions[0].label, 16)
        : gd.layout.xaxis,
      yaxis: state.droppedDimension
        ? { ...getProjectedAxis(keptDimensions[1].label, 24), scaleanchor: "x", scaleratio: 1 }
        : gd.layout.yaxis,
      dragmode: state.droppedDimension ? "pan" : gd.layout.dragmode,
    };

    Plotly.react(gd, data, layout, {
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
    }).then(() => {
      updateProjectionAttributes();
      renderYearControls();
      applyPlayerSearchHighlight();
      if (typeof window.applyButtonStyles === "function") window.applyButtonStyles();
      if (typeof window.adjustPlotlyForMobile === "function") window.adjustPlotlyForMobile();
    });
  }

  function getControlBar(wrapper) {
    let controlBar = wrapper.querySelector("#archetype-controls");
    if (controlBar) return controlBar;

    controlBar = document.createElement("div");
    controlBar.id = "archetype-controls";
    wrapper.insertBefore(controlBar, wrapper.firstChild);
    return controlBar;
  }

  function resizeGraph() {
    const gd = getGraph();
    if (!gd || !window.Plotly || !window.Plotly.Plots) return;
    window.requestAnimationFrame(() => window.Plotly.Plots.resize(gd));
  }

  function updateProjectionAttributes() {
    const wrapper = document.getElementById("plotly-wrapper");
    if (!wrapper) return;

    wrapper.dataset.projectionMode = state.droppedDimension ? "2d" : "3d";
    wrapper.dataset.droppedDimension = state.droppedDimension || "";
    wrapper.dataset.projectionReady = state.originalData ? "true" : "false";
  }

  function updateButtons() {
    document.querySelectorAll(".dimension-toggle-btn").forEach((button) => {
      const isShown = button.dataset.dimensionKey !== state.droppedDimension;
      button.classList.toggle("is-selected", isShown);
      button.setAttribute("aria-pressed", isShown ? "true" : "false");
      button.title = isShown
        ? `Hide ${button.dataset.dimensionLabel} from the plot`
        : `Show ${button.dataset.dimensionLabel} in the plot`;
    });
  }

  function updateYearButtons() {
    document.querySelectorAll(".year-toggle-btn").forEach((button) => {
      const isActive = Number(button.dataset.yearIndex) === state.activeYearIndex;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
  }

  function applyYearFilter(index, yearButton) {
    const gd = getGraph();
    if (!gd || !window.Plotly || !yearButton) return;

    state.activeYearIndex = index;
    updateYearButtons();

    const args = yearButton.args || [];
    Plotly.update(gd, args[0] || {}, args[1] || {}).then(() => {
      if (state.droppedDimension) applyProjection();
      applyPlayerSearchHighlight();
    });
  }

  function renderYearControls() {
    const wrapper = document.getElementById("plotly-wrapper");
    const gd = getGraph();
    if (!wrapper || !gd || wrapper.querySelector("#year-toggle")) return;

    const yearMenu = gd.layout && gd.layout.updatemenus && gd.layout.updatemenus[0];
    if (!yearMenu || !Array.isArray(yearMenu.buttons) || !yearMenu.buttons.length) return;

    const controls = document.createElement("div");
    controls.id = "year-toggle";

    const availableYearButtons = yearMenu.buttons
      .map((yearButton, index) => ({ yearButton, index }))
      .filter(({ yearButton }) => !isCurrentSeasonWindow || String(yearButton.label) === "2026");

    availableYearButtons.forEach(({ yearButton, index }) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "year-toggle-btn";
      button.dataset.yearIndex = String(index);
      button.textContent = yearButton.label;
      button.setAttribute("aria-pressed", index === state.activeYearIndex ? "true" : "false");
      button.addEventListener("click", () => applyYearFilter(index, yearButton));
      controls.appendChild(button);
    });

    getControlBar(wrapper).appendChild(controls);
    if (isCurrentSeasonWindow && availableYearButtons.length) {
      const currentSeason = availableYearButtons[0];
      applyYearFilter(currentSeason.index, currentSeason.yearButton);
      return;
    }
    updateYearButtons();
  }

  function getPlayerSearchOptions(gd) {
    if (!ensureOriginalData(gd)) return [];

    const players = new Set();
    state.originalData.forEach((traceData) => {
      traceData.hovertext.forEach((label) => {
        const playerName = playerNameFromHover(label);
        if (playerName) players.add(playerName);
      });
    });

    return Array.from(players).sort((a, b) => a.localeCompare(b));
  }

  function renderPlayerSearch() {
    const wrapper = document.getElementById("plotly-wrapper");
    const gd = getGraph();
    if (!wrapper || !gd || wrapper.querySelector("#player-search")) return;

    const controls = document.createElement("div");
    controls.id = "player-search";

    const input = document.createElement("input");
    input.type = "search";
    input.id = "player-search-input";
    input.setAttribute("list", "player-search-options");
    input.placeholder = "Search player";
    input.autocomplete = "off";
    input.value = state.playerSearch;
    input.addEventListener("input", () => {
      state.playerSearch = input.value;
      applyPlayerSearchHighlight();
    });

    const dataList = document.createElement("datalist");
    dataList.id = "player-search-options";
    getPlayerSearchOptions(gd).forEach((player) => {
      const option = document.createElement("option");
      option.value = player;
      dataList.appendChild(option);
    });

    const clearButton = document.createElement("button");
    clearButton.type = "button";
    clearButton.className = "player-search-clear";
    clearButton.textContent = "Clear";
    clearButton.addEventListener("click", () => {
      state.playerSearch = "";
      input.value = "";
      applyPlayerSearchHighlight();
      input.focus();
    });

    const status = document.createElement("span");
    status.id = "player-search-status";

    controls.appendChild(input);
    controls.appendChild(dataList);
    controls.appendChild(clearButton);
    controls.appendChild(status);
    getControlBar(wrapper).appendChild(controls);
    applyPlayerSearchHighlight();
  }

  function renderControls() {
    const wrapper = document.getElementById("plotly-wrapper");
    const gd = getGraph();
    if (!wrapper || !gd) return;

    if (!ensureOriginalData(gd)) return;
    updateProjectionAttributes();
    renderYearControls();
    renderPlayerSearch();

    const existingDimensionToggle = wrapper.querySelector("#dimension-toggle");
    if (existingDimensionToggle && !existingDimensionToggle.querySelector(".dimension-toggle-label")) {
      existingDimensionToggle.remove();
    } else if (existingDimensionToggle) {
      return;
    }

    const controls = document.createElement("div");
    controls.id = "dimension-toggle";

    const label = document.createElement("span");
    label.className = "dimension-toggle-label";
    label.textContent = "Show:";
    controls.appendChild(label);

    getDimensions(gd).forEach((dimension) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "dimension-toggle-btn";
      button.dataset.dimensionKey = dimension.key;
      button.dataset.dimensionLabel = dimension.label;
      button.textContent = dimension.label;
      button.title = "Hide " + dimension.label + " from the plot";
      button.setAttribute("aria-pressed", "true");
      button.addEventListener("click", () => {
        state.droppedDimension = state.droppedDimension === dimension.key ? null : dimension.key;
        updateButtons();
        applyProjection();
      });
      controls.appendChild(button);
    });

    getControlBar(wrapper).appendChild(controls);
    updateButtons();
    resizeGraph();
  }

  function injectStyles() {
    if (document.getElementById("projection-control-styles")) return;

    const style = document.createElement("style");
    style.id = "projection-control-styles";
    style.textContent = `
      body { margin: 0; }
      #plotly-wrapper {
        position: relative;
        min-height: 0;
      }
      #plotly-wrapper .plotly-graph-div {
        height: 100% !important;
        min-height: 0;
        padding-top: 112px;
        box-sizing: border-box;
      }
      #archetype-controls {
        position: absolute;
        top: 8px;
        left: 10px;
        right: 10px;
        z-index: 30;
        display: grid;
        grid-template-columns: minmax(0, max-content);
        align-items: start;
        justify-items: start;
        gap: 6px;
        pointer-events: none;
      }
      #plotly-wrapper .updatemenu-container {
        display: none !important;
      }
      #year-toggle {
        display: flex;
        max-width: 100%;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.34);
        border-radius: 999px;
        background: rgba(9, 14, 30, 0.78);
        box-shadow: 0 10px 26px rgba(4, 8, 18, 0.28);
        pointer-events: auto;
      }
      .year-toggle-btn {
        appearance: none;
        min-height: 34px;
        border: 0;
        border-right: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(17, 24, 46, 0.96);
        color: rgba(245, 247, 255, 0.84);
        cursor: pointer;
        font: 800 13px/1 "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        padding: 0 14px;
        white-space: nowrap;
        transition: background 0.18s ease, color 0.18s ease;
      }
      .year-toggle-btn:last-child {
        border-right: 0;
      }
      .year-toggle-btn:hover,
      .year-toggle-btn:focus-visible {
        background: rgba(0, 245, 138, 0.1);
        color: #00f58a;
        outline: none;
      }
      .year-toggle-btn.is-active {
        background: rgba(0, 245, 138, 0.2);
        color: #00f58a;
      }
      #dimension-toggle {
        position: static !important;
        top: auto !important;
        left: auto !important;
        display: flex;
        gap: 6px;
        flex-wrap: nowrap;
        max-width: 100%;
        padding: 5px;
        background: rgba(9, 14, 30, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 999px;
        box-shadow: 0 10px 26px rgba(4, 8, 18, 0.28);
        pointer-events: auto;
      }
      #player-search {
        display: flex;
        align-items: center;
        gap: 6px;
        max-width: 100%;
        padding: 5px;
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 999px;
        background: rgba(9, 14, 30, 0.76);
        box-shadow: 0 10px 26px rgba(4, 8, 18, 0.28);
        pointer-events: auto;
      }
      #player-search-input {
        appearance: none;
        width: 210px;
        min-height: 26px;
        border: 1px solid rgba(148, 163, 184, 0.36);
        border-radius: 999px;
        background: rgba(17, 24, 46, 0.94);
        color: #f8fafc;
        font: 800 10px/1.1 "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        outline: none;
        padding: 5px 11px;
      }
      #player-search-input::placeholder {
        color: rgba(245, 247, 255, 0.5);
      }
      #player-search-input:hover,
      #player-search-input:focus {
        border-color: #00f58a;
      }
      .player-search-clear {
        appearance: none;
        min-height: 26px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 999px;
        background: rgba(17, 24, 46, 0.94);
        color: rgba(245, 247, 255, 0.72);
        cursor: pointer;
        font: 800 8px/1.1 "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 0.08em;
        padding: 5px 8px;
        text-transform: uppercase;
      }
      .player-search-clear:hover,
      .player-search-clear:focus-visible {
        border-color: #00f58a;
        color: #00f58a;
        outline: none;
      }
      #player-search-status {
        min-width: 52px;
        color: rgba(245, 247, 255, 0.62);
        font: 800 8px/1.1 "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        text-transform: uppercase;
        white-space: nowrap;
      }
      .dimension-toggle-label {
        display: inline-flex;
        min-height: 26px;
        align-items: center;
        padding: 0 5px 0 7px;
        color: rgba(245, 247, 255, 0.72);
        font: 800 9px/1.1 "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        white-space: nowrap;
      }
      .dimension-toggle-btn {
        appearance: none;
        min-height: 26px;
        max-width: 148px;
        border: 1px solid rgba(148, 163, 184, 0.36);
        border-radius: 999px;
        background: rgba(17, 24, 46, 0.94);
        color: #f8fafc;
        cursor: pointer;
        font: 800 9px/1.1 "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 0.02em;
        padding: 5px 9px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
      }
      .dimension-toggle-btn:hover,
      .dimension-toggle-btn:focus-visible {
        border-color: #00f58a;
        background: rgba(0, 245, 138, 0.09);
        color: #00f58a;
        outline: none;
      }
      .dimension-toggle-btn:hover {
        transform: translateY(-1px);
      }
      .dimension-toggle-btn.is-selected {
        background: rgba(0, 245, 138, 0.16);
        border-color: #00f58a;
        color: #00f58a;
        box-shadow: inset 0 0 0 1px rgba(0, 245, 138, 0.16);
      }
      @media (max-width: 768px) {
        #plotly-wrapper .plotly-graph-div {
          padding-top: 108px;
        }
        #archetype-controls {
          gap: 5px;
          top: 6px;
          left: 4px;
          right: 58px;
        }
        #year-toggle {
          max-width: 100%;
        }
        .year-toggle-btn {
          min-height: 32px;
          padding: 0 10px;
          font-size: 11px;
        }
        #dimension-toggle {
          gap: 3px;
          width: fit-content;
          max-width: 100%;
          padding: 4px 3px;
        }
        .dimension-toggle-label {
          min-height: 28px;
          padding: 0 1px 0 3px;
          font-size: 6px;
        }
        .dimension-toggle-btn {
          flex: 0 0 auto;
          min-width: 0;
          max-width: none;
          padding: 5px 7px;
          overflow: visible;
          text-overflow: clip;
          font-size: 7px;
        }
        #player-search {
          max-width: 100%;
          margin-left: 0;
          padding: 4px;
        }
        #player-search-input {
          width: 150px;
          min-height: 28px;
          font-size: 8px;
        }
        .player-search-clear {
          min-height: 28px;
          font-size: 7px;
          padding: 5px 7px;
        }
        #player-search-status {
          min-width: 42px;
          font-size: 7px;
        }
      }
    `;
    document.head.appendChild(style);
  }

  function init() {
    injectStyles();
    renderControls();
    setTimeout(renderControls, 100);
    setTimeout(renderControls, 500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
