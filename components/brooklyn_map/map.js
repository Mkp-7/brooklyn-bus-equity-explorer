/* Brooklyn Bus Equity Explorer map component.
 *
 * The Leaflet map, its basemap tiles, the legend and the popups live here and are created once. Each time Streamlit sends new
 * data, only the tract, route and stop layers are replaced, so the basemap never reloads and the view is kept unless the
 * selection asks for a new one. Clicks and hovers are handled here in the browser and never rerun the app.
 *
 * Data arrives compactly: shapes as encoded polylines, labels as short strings, and the tooltip text is built here.
 */
(function () {
  "use strict";

  var S = {
    map: null, renderer: null, tile: null, tileKey: "", groups: { tracts: null, routes: null, stops: null },
    tip: null, popup: null, popupOpen: false, viewKey: null, height: 0, ctx: {}, renders: 0, last: null
  };
  window.__bb = S;                                    // handle used by the tests

  // ---------------------------------------------------------------- Streamlit component protocol
  function send(type, data) {
    var m = { isStreamlitMessage: true, type: type };
    for (var k in data) { m[k] = data[k]; }
    window.parent.postMessage(m, "*");
  }
  window.addEventListener("message", function (ev) {
    var d = ev.data;
    if (d && d.type === "streamlit:render") {
      try { render((d.args || {}).payload || null); } catch (e) { console.error("map render failed", e); }
    }
  });
  send("streamlit:componentReady", { apiVersion: 1 });

  // ---------------------------------------------------------------- helpers
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function num(n) { return Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 }); }

  // Google encoded polyline, 5 decimals: a string of lat/lon steps.
  function decode(str) {
    var i = 0, lat = 0, lng = 0, out = [], b, shift, result;
    while (i < str.length) {
      shift = 0; result = 0;
      do { b = str.charCodeAt(i++) - 63; result |= (b & 31) << shift; shift += 5; } while (b >= 32);
      lat += (result & 1) ? ~(result >> 1) : (result >> 1);
      shift = 0; result = 0;
      do { b = str.charCodeAt(i++) - 63; result |= (b & 31) << shift; shift += 5; } while (b >= 32);
      lng += (result & 1) ? ~(result >> 1) : (result >> 1);
      out.push([lat / 1e5, lng / 1e5]);
    }
    return out;
  }

  function row(k, v) { return "<div class='tip-row'><span>" + k + "</span><b>" + v + "</b></div>"; }
  function head(t) { return "<div class='tip'><div class='tip-title'>" + t + "</div>"; }

  function html(layer) {
    var d = layer._d, c = S.ctx;
    if (layer._k === "tract") {
      return head(esc(c.unit) + " " + esc(d[0])) + row("Equity score", esc(d[7])) + row("Residents", esc(d[8])) +
        row("Minority", esc(d[9])) + row("Poverty", esc(d[10])) + row("Nearest " + esc(c.scope), esc(d[11])) +
        row("Within ¼ mile", esc(d[12])) + "</div>";
    }
    if (layer._k === "route") {
      return head("Route " + esc(d[0])) + row("Boardings", num(d[2])) + row("Alightings", num(d[3])) + row("Date", esc(c.period)) +
        row("Expected wait, AM peak", esc(d[6])) + row("Expected wait, PM peak", esc(d[7])) + row("Scheduled trips", num(d[4])) +
        row("Stops", num(d[5])) + (c.direction ? row("Direction", esc(c.direction)) : "") + "</div>";
    }
    return head(esc(d[3])) + row("Stop ID", esc(d[4])) + row("Boardings", num(d[5])) + row("Alightings", num(d[6])) +
      row("Date", esc(c.period)) + row("Routes", esc(d[7])) + row("Expected wait", esc(d[8])) + "</div>";
  }

  // ---------------------------------------------------------------- the layers
  var HOVER = { tract: { color: "#ffd400", opacity: 1, weight: 3 }, route: { color: "#ffd400", opacity: 1, weight: 6 },
                stop: { color: "#ffd400", opacity: 1, weight: 3 } };

  function buildTracts(rows) {
    var out = new Array(rows.length);
    for (var i = 0; i < rows.length; i++) {
      var d = rows[i];
      var p = L.polygon(decode(d[1]), { renderer: S.renderer, fill: true, fillColor: d[2], fillOpacity: d[3],
                                        color: d[4], opacity: d[5], weight: d[6] });
      p._d = d; p._k = "tract"; p._base = { color: d[4], opacity: d[5], weight: d[6] };
      out[i] = p;
    }
    return out;
  }
  function buildRoutes(rows) {
    var out = new Array(rows.length);
    for (var i = 0; i < rows.length; i++) {
      var d = rows[i];
      var p = L.polyline(decode(d[1]), { renderer: S.renderer, color: "#ee7020", weight: 3, opacity: 0.85 });
      p._d = d; p._k = "route"; p._base = { color: "#ee7020", opacity: 0.85, weight: 3 };
      out[i] = p;
    }
    return out;
  }
  function buildStops(rows) {
    var out = new Array(rows.length);
    for (var i = 0; i < rows.length; i++) {
      var d = rows[i];
      var p = L.circleMarker([d[1], d[0]], { renderer: S.renderer, radius: d[2], fillColor: "#0f5257", fillOpacity: 0.95,
                                             color: "#ffffff", opacity: 1, weight: 1.5 });
      p._d = d; p._k = "stop"; p._base = { color: "#ffffff", opacity: 1, weight: 1.5 };
      out[i] = p;
    }
    return out;
  }

  function wire(group) {
    group.on("mouseover", function (e) {
      if (S.popupOpen) { return; }                       // while a popup is open, no other tooltip shows
      e.layer.setStyle(HOVER[e.layer._k]);
      S.tip.setContent(html(e.layer)).setLatLng(e.latlng);
      S.map.openTooltip(S.tip);
    });
    group.on("mousemove", function (e) { if (!S.popupOpen && S.map.hasLayer(S.tip)) { S.tip.setLatLng(e.latlng); } });
    group.on("mouseout", function (e) { e.layer.setStyle(e.layer._base); S.map.closeTooltip(S.tip); });
    group.on("click", function (e) {
      S.map.closeTooltip(S.tip);
      S.popup.setLatLng(e.latlng).setContent(html(e.layer));
      S.map.openPopup(S.popup);
    });
  }

  function replace(name, layers) {
    var old = S.groups[name];
    if (old) { old.clearLayers(); S.map.removeLayer(old); S.groups[name] = null; }
    if (!layers || !layers.length) { return; }
    var g = L.featureGroup(layers);
    wire(g);
    S.groups[name] = g;
  }

  // ---------------------------------------------------------------- one-time setup
  function init(p) {
    var useCanvas = L.Browser.canvas && p.renderer !== "svg";
    S.renderer = useCanvas ? L.canvas({ padding: 0.4, tolerance: 4 }) : L.svg({ padding: 0.4 });
    S.map = L.map("map", { preferCanvas: useCanvas, zoomSnap: 0.25, zoomControl: true, attributionControl: true });
    var v = p.view || {};
    if (v.bounds) { S.map.fitBounds(v.bounds, { padding: [20, 20], animate: false }); }
    else { S.map.setView(v.center || [40.65, -73.95], v.zoom || 11, { animate: false }); }
    S.viewKey = p.viewKey;
    S.tip = L.tooltip({ direction: "right", offset: [14, 0], opacity: 1 });
    S.popup = L.popup({ maxWidth: 360, autoPan: true });
    S.map.on("popupopen", function () {
      S.popupOpen = true; S.map.closeTooltip(S.tip);
      S.map.getContainer().classList.add("popup-open");
    });
    S.map.on("popupclose", function () { S.popupOpen = false; S.map.getContainer().classList.remove("popup-open"); });
  }

  // ---------------------------------------------------------------- every update from Streamlit
  function render(p) {
    if (!p) { return; }
    var t0 = (window.performance || Date).now();
    var first = !S.map;
    var mapEl = document.getElementById("map");
    if (p.height && p.height !== S.height) {
      S.height = p.height; mapEl.style.height = p.height + "px";
      send("streamlit:setFrameHeight", { height: p.height });
    }
    if (first) { init(p); } else if (p.height) { S.map.invalidateSize(); }
    S.ctx = p.ctx || {};

    // basemap tiles: created once, replaced only if the setting changes
    var tk = JSON.stringify(p.tiles || null);
    if (tk !== S.tileKey) {
      if (S.tile) { S.map.removeLayer(S.tile); S.tile = null; }
      if (p.tiles) {
        S.tile = L.tileLayer(p.tiles.url, { attribution: p.tiles.attr, minZoom: p.tiles.minZoom, maxZoom: p.tiles.maxZoom,
                                            maxNativeZoom: p.tiles.maxNative });
        S.tile.addTo(S.map);
      }
      S.tileKey = tk;
    }

    // the data layers: replaced, drawn from the bottom up
    S.map.closePopup();
    replace("tracts", buildTracts(p.tracts || []));
    replace("routes", buildRoutes(p.routes || []));
    replace("stops", buildStops(p.stops || []));
    ["tracts", "routes", "stops"].forEach(function (n) { if (S.groups[n]) { S.groups[n].addTo(S.map); } });

    // the view changes only when the selection asks for it, so panning and zooming survive a new date or list
    if (!first && p.viewKey !== S.viewKey) {
      S.viewKey = p.viewKey;
      var v = p.view || {};
      if (v.bounds) { S.map.fitBounds(v.bounds, { padding: [20, 20] }); }
      else if (v.center) { S.map.setView(v.center, v.zoom || 14); }
    }

    // legend and note
    var lg = p.legend, el = document.getElementById("legend");
    if (lg) {
      el.style.display = "";
      el.innerHTML = "<div class='lg-t'>Need for bus service</div><div class='lg-bar' style='background:linear-gradient(90deg,rgb(" +
        lg.loRgb.join(",") + "),rgb(" + lg.hiRgb.join(",") + "))'></div><div class='lg-sc'><span>" + Math.round(lg.lo) + "</span><span>" +
        Math.round(lg.hi) + "</span></div><div class='lg-r'><span class='lg-line'></span>Route</div>" +
        "<div class='lg-r'><span class='lg-dot'></span>Stop, sized by boardings</div><div class='lg-r'><span class='lg-gray'></span>Not scored</div>";
    } else { el.style.display = "none"; }
    var nt = document.getElementById("note");
    if (p.notes && p.notes.length) { nt.textContent = p.notes.join(" "); nt.style.display = "block"; } else { nt.style.display = "none"; }

    S.renders += 1;
    S.last = { ms: (window.performance || Date).now() - t0, tracts: (p.tracts || []).length, routes: (p.routes || []).length,
               stops: (p.stops || []).length };
  }
})();
