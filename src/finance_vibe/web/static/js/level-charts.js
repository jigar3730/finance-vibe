// Level charts: recent candles with a card's key levels as price lines.
// Each <div class="level-chart"> carries data-mode, data-symbol and data-levels, a
// JSON list of {price, label, color, style} built by playbook.range_levels (style is
// a LightweightCharts LineStyle: 0 solid, 1 dotted, 2 dashed). Bars come from
// /api/bars/<mode>/<symbol>. Needs the vendored LightweightCharts global (v4).
(function () {
  "use strict";

  var COLORS = {
    text: "#787b86",
    grid: "#2a2e39",
    up: "#26a69a",
    down: "#f23645",
  };

  function levelValues(el) {
    var levels;
    try {
      levels = JSON.parse(el.dataset.levels || "[]");
    } catch (err) {
      return [];
    }
    return levels.filter(function (level) {
      return isFinite(level.price);
    });
  }

  function showMessage(el, text) {
    el.textContent = text;
    el.classList.add("level-chart-empty");
  }

  function draw(el, bars) {
    var levels = levelValues(el);
    var chart = LightweightCharts.createChart(el, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: COLORS.text, fontSize: 11 },
      grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
      rightPriceScale: { borderColor: COLORS.grid },
      timeScale: { borderColor: COLORS.grid },
      handleScroll: false,
      handleScale: false,
    });
    var series = chart.addCandlestickSeries({
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderVisible: false,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
      // Keep every level on screen, not just the candles' range.
      autoscaleInfoProvider: function (original) {
        var range = original();
        if (!range || !levels.length) return range;
        var values = levels.map(function (l) { return l.price; });
        return {
          priceRange: {
            minValue: Math.min(range.priceRange.minValue, Math.min.apply(null, values)),
            maxValue: Math.max(range.priceRange.maxValue, Math.max.apply(null, values)),
          },
        };
      },
    });
    series.setData(bars);
    levels.forEach(function (level) {
      series.createPriceLine({
        price: level.price,
        color: level.color,
        lineWidth: 1,
        lineStyle: level.style,
        axisLabelVisible: true,
        title: level.label,
      });
    });
    chart.timeScale().fitContent();
  }

  function load(el) {
    var url = "/api/bars/" + encodeURIComponent(el.dataset.mode) + "/" + encodeURIComponent(el.dataset.symbol);
    fetch(url)
      .then(function (response) {
        if (!response.ok) throw new Error(response.status);
        return response.json();
      })
      .then(function (bars) {
        if (!bars.length) return showMessage(el, "No bars available.");
        draw(el, bars);
      })
      .catch(function () {
        showMessage(el, "Chart unavailable.");
      });
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (typeof LightweightCharts === "undefined") return;
    document.querySelectorAll(".level-chart").forEach(load);
  });
})();
