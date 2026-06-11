/* Ruqola server docs — client-side tabbed viewer.
   The Markdown files under docs/, gpuq/, examples/ are the source of truth;
   this script fetches and renders them into a tabbed, sidebar-navigated site.
   No build step: works as static files on GitHub Pages (.nojekyll). */
(function () {
  "use strict";

  // ---- Content map: tabs -> docs (paths relative to repo root) ----
  var MANIFEST = [
    { id: "overview", label: "Overview", docs: [
      { title: "Overview", path: "docs/overview.md" },
    ]},
    { id: "start", label: "Getting Started", docs: [
      { title: "Bash Basics", path: "docs/bash-basics.md" },
      { title: "Best Practices", path: "docs/best-practices.md" },
      { title: "Scratch Storage", path: "docs/scratch-folder.md" },
    ]},
    { id: "gpuq", label: "GPU Queue (gpuq)", docs: [
      { title: "gpuq User Guide", path: "docs/gpu-queue-guide.md" },
      { title: "Notifications FAQ", path: "docs/notifications-faq.md" },
      { title: "gpuq Reference", path: "gpuq/README.md" },
    ]},
    { id: "frameworks", label: "Frameworks", docs: [
      { title: "PyTorch", path: "docs/pytorch-guide.md" },
      { title: "TensorFlow", path: "docs/tensorflow-guide.md" },
      { title: "JAX", path: "docs/jax-guide.md" },
      { title: "Transformers", path: "docs/transformers-guide.md" },
      { title: "Examples", path: "examples/README.md" },
    ]},
    { id: "hardware", label: "Hardware", docs: [
      { title: "H200 Specs", path: "docs/h200-specs.md" },
    ]},
    { id: "hopper", label: "Hopper (NUS)", docs: [
      { title: "Hopper Access", path: "docs/hopper.md" },
    ]},
    // Admin tab temporarily hidden from the site (docs remain in the repo under
    // docs/users-creation.md and docs/users-quota.md). Re-enable by uncommenting.
    // { id: "admin", label: "Admin", docs: [
    //   { title: "User Creation", path: "docs/users-creation.md" },
    //   { title: "User Quotas", path: "docs/users-quota.md" },
    // ]},
    { id: "help", label: "Troubleshooting", docs: [
      { title: "Troubleshooting", path: "docs/troubleshooting.md" },
    ]},
  ];

  // ---- Lookups ----
  var byTab = {};
  var allDocs = []; // {tabId, slug, title, path}
  MANIFEST.forEach(function (tab) {
    byTab[tab.id] = tab;
    tab.docs.forEach(function (d) {
      d.slug = slugFromPath(d.path);
      d.tabId = tab.id;
      allDocs.push(d);
    });
  });

  function slugFromPath(p) {
    var base = p.split("/").pop().replace(/\.md$/i, "");
    if (base.toLowerCase() === "readme") {
      // disambiguate README.md by its directory (gpuq, examples, ...)
      var parts = p.split("/");
      base = parts.length > 1 ? parts[parts.length - 2] + "-readme" : "readme";
    }
    return base.toLowerCase();
  }

  // Resolve an in-markdown href to an internal route, or null if external/unknown.
  function resolveInternal(href) {
    if (!href) return null;
    if (/^(https?:|mailto:|tel:|#)/i.test(href)) return null;
    var clean = href.split("#")[0].split("?")[0];
    clean = clean.replace(/^\.\//, "").replace(/(\.\.\/)+/g, "");
    if (!/\.md$/i.test(clean)) return null;
    // 1) exact path or suffix match
    var hit = allDocs.find(function (d) { return d.path === clean || d.path.toLowerCase().endsWith("/" + clean.toLowerCase()) || d.path.toLowerCase() === clean.toLowerCase(); });
    // 2) basename match (for bare "foo.md" links inside docs/)
    if (!hit) {
      var base = clean.split("/").pop().toLowerCase();
      var cands = allDocs.filter(function (d) { return d.path.split("/").pop().toLowerCase() === base; });
      // prefer a docs/ candidate if the link looks like a docs link
      hit = cands.length === 1 ? cands[0] : cands.find(function (d) { return /^docs\//.test(d.path); }) || cands[0];
    }
    return hit ? { tabId: hit.tabId, slug: hit.slug } : null;
  }

  function slugify(text) {
    return String(text).toLowerCase().trim()
      .replace(/[^\w\s-]/g, "").replace(/\s+/g, "-").replace(/-+/g, "-");
  }

  // ---- Routing: #<tabId> or #<tabId>/<slug> ----
  function parseHash() {
    var h = (location.hash || "").replace(/^#\/?/, "");
    var parts = h.split("/").filter(Boolean);
    return { tabId: parts[0] || null, slug: parts[1] || null };
  }
  function routeFor(tabId, slug) { return "#" + tabId + (slug ? "/" + slug : ""); }

  // ---- Elements ----
  var elTabs = document.getElementById("tabs");
  var elLayout = document.querySelector(".layout");
  var elSidebar = document.getElementById("sidebar");
  var elContent = document.getElementById("content");
  var elToc = document.getElementById("toc");
  var elSearch = document.getElementById("search");
  var elResults = document.getElementById("search-results");
  var elTheme = document.getElementById("theme");

  // ---- Markdown config ----
  marked.setOptions({
    gfm: true, breaks: false,
    highlight: function (code, lang) {
      try {
        if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
        return hljs.highlightAuto(code).value;
      } catch (e) { return code; }
    },
  });

  var cache = {};
  function fetchDoc(path) {
    if (cache[path]) return Promise.resolve(cache[path]);
    return fetch(path, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(r.status + " " + r.statusText);
      return r.text();
    }).then(function (t) { cache[path] = t; return t; });
  }

  // ---- Render tab bar ----
  function renderTabs(activeTab) {
    elTabs.innerHTML = "";
    MANIFEST.forEach(function (tab) {
      var b = document.createElement("button");
      b.textContent = tab.label;
      b.className = tab.id === activeTab ? "active" : "";
      b.onclick = function () { navigate(tab.id, tab.docs[0].slug); };
      elTabs.appendChild(b);
    });
  }

  // ---- Render sidebar for a tab ----
  function renderSidebar(tab, activeSlug) {
    elSidebar.innerHTML = "";
    // Single-doc tabs hide the sidebar; tell the grid to drop its column so the
    // content doesn't auto-flow into the now-empty sidebar track.
    if (tab.docs.length <= 1) {
      elSidebar.style.display = "none";
      elLayout.classList.add("no-sidebar");
      return;
    }
    elSidebar.style.display = "";
    elLayout.classList.remove("no-sidebar");
    var h = document.createElement("h3");
    h.textContent = tab.label;
    elSidebar.appendChild(h);
    tab.docs.forEach(function (d) {
      var a = document.createElement("a");
      a.textContent = d.title;
      a.href = routeFor(tab.id, d.slug);
      a.className = d.slug === activeSlug ? "active" : "";
      a.onclick = function (e) { e.preventDefault(); navigate(tab.id, d.slug); };
      elSidebar.appendChild(a);
    });
  }

  // ---- Render a document ----
  function renderDoc(doc) {
    elContent.innerHTML = '<div class="loading">Loading ' + doc.title + '…</div>';
    elToc.innerHTML = "";
    fetchDoc(doc.path).then(function (md) {
      var html = marked.parse(md);
      elContent.innerHTML = '<div class="banner">📄 Source: <code>' + doc.path +
        '</code> — edit the Markdown to update this page.</div>' + html;
      postProcess(doc);
      window.scrollTo(0, 0);
    }).catch(function (err) {
      elContent.innerHTML = '<div class="errbox"><strong>Could not load <code>' + doc.path +
        '</code></strong><br>' + err.message +
        '<br><br>If you are viewing this from <code>file://</code>, browsers block <code>fetch()</code>. ' +
        'Serve the folder instead, e.g. <code>python3 -m http.server</code>, or view it via GitHub Pages.</div>';
    });
  }

  function postProcess(doc) {
    // 1) rewrite internal .md links to routes; mark external links
    elContent.querySelectorAll("a[href]").forEach(function (a) {
      var href = a.getAttribute("href");
      if (/^#/.test(href)) { // in-page anchor
        a.onclick = function (e) {
          e.preventDefault();
          var t = elContent.querySelector(href) || document.getElementById(href.slice(1));
          if (t) t.scrollIntoView({ behavior: "smooth" });
        };
        return;
      }
      var r = resolveInternal(href);
      if (r) {
        a.setAttribute("href", routeFor(r.tabId, r.slug));
        a.onclick = function (e) { e.preventDefault(); navigate(r.tabId, r.slug); };
      } else if (/^https?:/i.test(href)) {
        a.classList.add("ext"); a.target = "_blank"; a.rel = "noopener";
      }
    });
    // 2) heading ids + TOC
    var heads = elContent.querySelectorAll("h2, h3");
    var used = {};
    var tocItems = [];
    heads.forEach(function (h) {
      var id = slugify(h.textContent);
      if (used[id]) { used[id]++; id = id + "-" + used[id]; } else { used[id] = 1; }
      h.id = id;
      tocItems.push({ id: id, text: h.textContent, level: h.tagName === "H3" ? 3 : 2 });
    });
    if (tocItems.length > 1) {
      var frag = '<h4>On this page</h4>';
      tocItems.forEach(function (it) {
        frag += '<a href="#' + it.id + '" class="lvl-' + it.level + '" data-id="' + it.id + '">' + escapeHtml(it.text) + '</a>';
      });
      elToc.innerHTML = frag;
      elToc.querySelectorAll("a").forEach(function (a) {
        a.onclick = function (e) {
          e.preventDefault();
          var t = document.getElementById(a.getAttribute("data-id"));
          if (t) t.scrollIntoView({ behavior: "smooth" });
        };
      });
      setupScrollSpy(tocItems);
    }
    // 3) copy buttons on code blocks
    elContent.querySelectorAll("pre").forEach(function (pre) {
      var btn = document.createElement("button");
      btn.className = "copy-btn"; btn.textContent = "Copy";
      btn.onclick = function () {
        var code = pre.querySelector("code");
        var text = code ? code.innerText : pre.innerText;
        navigator.clipboard.writeText(text).then(function () {
          btn.textContent = "Copied!"; btn.classList.add("copied");
          setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1500);
        });
      };
      pre.appendChild(btn);
    });
  }

  var scrollSpy = null;
  function setupScrollSpy(items) {
    if (scrollSpy) window.removeEventListener("scroll", scrollSpy);
    scrollSpy = function () {
      var pos = window.scrollY + 90;
      var current = items[0] && items[0].id;
      for (var i = 0; i < items.length; i++) {
        var el = document.getElementById(items[i].id);
        if (el && el.offsetTop <= pos) current = items[i].id;
      }
      elToc.querySelectorAll("a").forEach(function (a) {
        a.classList.toggle("active", a.getAttribute("data-id") === current);
      });
    };
    window.addEventListener("scroll", scrollSpy, { passive: true });
    scrollSpy();
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  // ---- Navigation ----
  function navigate(tabId, slug) {
    var target = routeFor(tabId, slug);
    if (location.hash !== target) { location.hash = target; }
    else { render(); } // same hash: force render (e.g., first load)
  }

  function render() {
    var route = parseHash();
    var tab = byTab[route.tabId] || MANIFEST[0];
    var doc = tab.docs.find(function (d) { return d.slug === route.slug; }) || tab.docs[0];
    renderTabs(tab.id);
    renderSidebar(tab, doc.slug);
    renderDoc(doc);
  }

  // ---- Search (title match across all docs) ----
  function runSearch(q) {
    q = q.trim().toLowerCase();
    if (!q) { elResults.classList.remove("show"); elResults.innerHTML = ""; return; }
    var matches = allDocs.filter(function (d) {
      return d.title.toLowerCase().indexOf(q) >= 0 || byTab[d.tabId].label.toLowerCase().indexOf(q) >= 0;
    }).slice(0, 12);
    if (!matches.length) {
      elResults.innerHTML = '<a><small>No matching pages</small></a>';
    } else {
      elResults.innerHTML = matches.map(function (d) {
        return '<a href="' + routeFor(d.tabId, d.slug) + '" data-tab="' + d.tabId + '" data-slug="' + d.slug + '">' +
          escapeHtml(d.title) + ' <small>· ' + escapeHtml(byTab[d.tabId].label) + '</small></a>';
      }).join("");
      elResults.querySelectorAll("a").forEach(function (a) {
        a.onclick = function (e) {
          e.preventDefault();
          navigate(a.getAttribute("data-tab"), a.getAttribute("data-slug"));
          elSearch.value = ""; elResults.classList.remove("show");
        };
      });
    }
    elResults.classList.add("show");
  }

  // ---- Theme ----
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    document.getElementById("hl-light").disabled = (t === "dark");
    document.getElementById("hl-dark").disabled = (t !== "dark");
    elTheme.textContent = t === "dark" ? "☀" : "☾";
    try { localStorage.setItem("ruqola-theme", t); } catch (e) {}
  }
  function initTheme() {
    var saved;
    try { saved = localStorage.getItem("ruqola-theme"); } catch (e) {}
    if (!saved) saved = (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light";
    applyTheme(saved);
    elTheme.onclick = function () {
      applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
    };
  }

  // ---- Boot ----
  initTheme();
  elSearch.addEventListener("input", function () { runSearch(elSearch.value); });
  elSearch.addEventListener("focus", function () { if (elSearch.value) runSearch(elSearch.value); });
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".search-wrap")) elResults.classList.remove("show");
  });
  window.addEventListener("hashchange", render);
  if (!location.hash) location.hash = routeFor("overview", "overview");
  render();
})();
