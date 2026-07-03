// In-page helper library injected into every document via CDP
// (Page.addScriptToEvaluateOnNewDocument + a one-shot Runtime.evaluate).
//
// It defines `window.__openagentChrome` with the DOM-side primitives the
// structured tools need: an accessibility-tree renderer, natural-language
// element finder, form setter, article-text extractor, and a stable element
// ref map (ref_N ↔ element) backed by WeakRef so refs survive across tool
// calls without leaking. mcp-server.js calls these through Runtime.evaluate.
//
// This is a self-contained IIFE with no extension APIs — it runs as ordinary
// page script under CDP. Ported from the previous content.js.

export const PAGE_SCRIPT = String.raw`
(function () {
  if (window.__openagentChromeLoaded) return;
  window.__openagentChromeLoaded = true;

  let refCounter = 0;
  const elementMap = {};
  const reverseMap = new WeakMap();

  function getOrAssignRef(el) {
    const existing = reverseMap.get(el);
    if (existing && elementMap[existing] && elementMap[existing].deref() === el) return existing;
    const ref = "ref_" + (++refCounter);
    elementMap[ref] = new WeakRef(el);
    reverseMap.set(el, ref);
    return ref;
  }

  function resolveRef(refId) {
    const wr = elementMap[refId];
    if (!wr) return null;
    const el = wr.deref();
    if (!el) { delete elementMap[refId]; return null; }
    return el;
  }

  const TAG_TO_ROLE = {
    a: "link", button: "button", input: "textbox", textarea: "textbox",
    select: "combobox", img: "img", h1: "heading", h2: "heading", h3: "heading",
    h4: "heading", h5: "heading", h6: "heading", nav: "navigation", main: "main",
    header: "banner", footer: "contentinfo", aside: "complementary", form: "form",
    table: "table", tr: "row", th: "columnheader", td: "cell", ul: "list",
    ol: "list", li: "listitem", dialog: "dialog", details: "group",
    summary: "button", progress: "progressbar", meter: "meter", video: "video",
    audio: "audio", section: "region", article: "article",
  };

  function getRole(el) {
    if (el.getAttribute("role")) return el.getAttribute("role");
    const tag = el.tagName.toLowerCase();
    if (tag === "input") {
      const type = (el.type || "text").toLowerCase();
      const typeRoles = {
        checkbox: "checkbox", radio: "radio", range: "slider", button: "button",
        submit: "button", reset: "button", search: "searchbox", number: "spinbutton",
      };
      return typeRoles[type] || "textbox";
    }
    return TAG_TO_ROLE[tag] || null;
  }

  function getAccessibleName(el) {
    const ariaLabel = el.getAttribute("aria-label");
    if (ariaLabel) return ariaLabel.trim();

    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const names = labelledBy.split(/\s+/)
        .map(function (id) { const n = document.getElementById(id); return n && n.textContent ? n.textContent.trim() : null; })
        .filter(Boolean);
      if (names.length) return names.join(" ");
    }

    if (el.placeholder) return el.placeholder.trim();
    if (el.title) return el.title.trim();
    if (el.alt) return el.alt.trim();

    if (el.id) {
      const label = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (label) return label.textContent.trim();
    }
    if (el.closest("label")) {
      const labelText = el.closest("label").textContent.trim();
      if (labelText) return labelText;
    }

    const tag = el.tagName.toLowerCase();
    if (["a","button","h1","h2","h3","h4","h5","h6","li","summary","label","th","td","span"].includes(tag)) {
      const text = el.textContent ? el.textContent.trim() : "";
      if (text && text.length < 200) return text;
    }
    return "";
  }

  function isInteractive(el) {
    const tag = el.tagName.toLowerCase();
    if (["a","button","input","textarea","select","summary","details"].includes(tag)) return true;
    const role = el.getAttribute("role");
    if (role && ["button","link","textbox","checkbox","radio","tab","menuitem","switch","combobox","slider","spinbutton","searchbox","option"].includes(role)) return true;
    if (el.tabIndex >= 0) return true;
    if (el.onclick || el.getAttribute("onclick")) return true;
    if (el.contentEditable === "true") return true;
    return false;
  }

  function isVisible(el) {
    if (el.offsetParent === null && el.tagName.toLowerCase() !== "body" && getComputedStyle(el).position !== "fixed") return false;
    const style = getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden") return false;
    return true;
  }

  function generateAccessibilityTree(options) {
    options = options || {};
    const filter = options.filter || "all";
    const maxDepth = options.depth || 15;
    const maxChars = options.max_chars || 50000;
    const startRefId = options.ref_id || null;

    let output = "";
    let charCount = 0;
    let truncated = false;

    function append(text) {
      if (truncated) return false;
      if (charCount + text.length > maxChars) {
        output += text.substring(0, maxChars - charCount);
        output += "\n... (truncated)";
        truncated = true;
        return false;
      }
      output += text;
      charCount += text.length;
      return true;
    }

    function walk(el, depth, indent) {
      if (truncated) return;
      if (depth > maxDepth) return;
      if (!el || el.nodeType !== 1) return;

      const tag = el.tagName.toLowerCase();
      if (["script","style","noscript","template"].includes(tag)) return;

      const role = getRole(el);
      const name = getAccessibleName(el);
      const interactive = isInteractive(el);
      const visible = isVisible(el);

      const isContainer = el.children.length > 0;
      if (filter === "interactive" && !interactive && !isContainer) return;

      const shouldShow =
        (filter === "all" && (role || name)) ||
        (filter === "interactive" && interactive);

      if (shouldShow && visible) {
        const ref = getOrAssignRef(el);
        let line = indent;
        if (role) line += role;
        if (name) line += ' "' + name.substring(0, 100) + '"';
        line += " [" + ref + "]";
        if (tag === "a" && el.href) line += ' href="' + el.href + '"';
        if (tag === "img" && el.src) line += ' src="' + el.src.substring(0, 100) + '"';
        if (["input","textarea"].includes(tag) && el.value) line += ' value="' + el.value.substring(0, 100) + '"';
        if (tag === "input") line += ' type="' + (el.type || "text") + '"';
        if (el.getAttribute("aria-expanded")) line += " expanded=" + el.getAttribute("aria-expanded");
        if (el.getAttribute("aria-checked")) line += " checked=" + el.getAttribute("aria-checked");
        if (el.getAttribute("aria-selected")) line += " selected=" + el.getAttribute("aria-selected");
        if (el.disabled) line += " disabled";
        if (tag === "select") {
          const opts = Array.from(el.options).map(function (o) { return (o.selected ? "*" : " ") + o.value + '="' + o.textContent.trim() + '"'; });
          if (opts.length) line += " options=[" + opts.join(", ") + "]";
        }
        if (!append(line + "\n")) return;
      }

      const nextIndent = shouldShow && visible ? indent + "  " : indent;
      if (el.shadowRoot) {
        for (const child of el.shadowRoot.children) walk(child, depth + 1, nextIndent);
      }
      for (const child of el.children) walk(child, depth + 1, nextIndent);
    }

    let root = document.body;
    if (startRefId) {
      const el = resolveRef(startRefId);
      if (el) root = el;
      else return 'Error: ref_id "' + startRefId + '" not found or element was garbage collected.';
    }
    if (!root) return "";
    walk(root, 0, "");
    return output;
  }

  // Collect visible text while descending into OPEN shadow roots (modern
  // sites — e.g. new Reddit's shreddit/faceplate web components — render posts
  // and comments inside shadow DOM, which textContent alone would miss).
  function deepText(root) {
    let out = "";
    const skip = new Set(["script", "style", "noscript", "template", "svg"]);
    const walk = (node) => {
      if (node.nodeType === 3) { out += node.nodeValue + " "; return; }
      if (node.nodeType !== 1) return;
      const tag = node.tagName ? node.tagName.toLowerCase() : "";
      if (skip.has(tag)) return;
      if (node.shadowRoot) for (const c of node.shadowRoot.childNodes) walk(c);
      for (const c of node.childNodes) walk(c);
    };
    walk(root);
    return out;
  }

  function getPageText() {
    const selectors = ["article","main",'[class*="articleBody"]','[class*="post-content"]','[class*="entry-content"]','[role="main"]',"shreddit-post",".content","#content"];
    let source = null;
    for (const sel of selectors) { source = document.querySelector(sel); if (source) break; }
    if (!source) source = document.body;

    const title = document.title || "";
    const url = location.href;
    const tag = source ? source.tagName.toLowerCase() : "body";

    let text = deepText(source).replace(/\s+/g, " ").trim();
    // Fallback: if the chosen container yielded almost nothing (heavily
    // shadow-slotted layouts), read the whole body deeply.
    if (text.length < 40 && source !== document.body) {
      text = deepText(document.body).replace(/\s+/g, " ").trim();
    }
    return JSON.stringify({ title: title, url: url, sourceTag: tag, text: text.substring(0, 100000) });
  }

  function findElements(query) {
    const q = String(query || "").toLowerCase();
    const results = [];

    function collectAll(root) {
      const elements = [];
      for (const el of root.querySelectorAll("*")) {
        elements.push(el);
        if (el.shadowRoot) elements.push.apply(elements, collectAll(el.shadowRoot));
      }
      return elements;
    }

    const all = collectAll(document);
    for (const el of all) {
      if (results.length >= 20) break;
      if (!isVisible(el)) continue;
      const tag = el.tagName.toLowerCase();
      if (["script","style","noscript","template"].includes(tag)) continue;

      const role = getRole(el) || "";
      const name = getAccessibleName(el) || "";
      const text = (el.textContent ? el.textContent.trim() : "").substring(0, 200);
      const placeholder = el.placeholder || "";
      const ariaLabel = el.getAttribute("aria-label") || "";
      const title = el.title || "";
      const type = el.type || "";

      const searchable = (role + " " + name + " " + text + " " + placeholder + " " + ariaLabel + " " + title + " " + type + " " + tag).toLowerCase();
      if (searchable.includes(q)) {
        const ref = getOrAssignRef(el);
        const rect = el.getBoundingClientRect();
        results.push({
          ref: ref,
          role: role || tag,
          name: name || text.substring(0, 80),
          coordinates: [Math.round(rect.x + rect.width / 2), Math.round(rect.y + rect.height / 2)],
        });
      }
    }
    return results;
  }

  function findInputInside(el) {
    const tag = el.tagName.toLowerCase();
    if (["input","textarea","select"].includes(tag)) return el;
    const root = el.shadowRoot || el;
    const inner = root.querySelector("input, textarea, select");
    if (inner) return inner;
    for (const child of root.querySelectorAll("*")) {
      if (child.shadowRoot) {
        const deep = child.shadowRoot.querySelector("input, textarea, select");
        if (deep) return deep;
      }
    }
    return null;
  }

  function setFormValue(refId, value) {
    const el = resolveRef(refId);
    if (!el) return { error: "Element " + refId + " not found or was garbage collected." };

    el.scrollIntoView({ block: "center", behavior: "instant" });
    const target = findInputInside(el) || el;
    const tag = target.tagName.toLowerCase();
    const type = (target.type || "").toLowerCase();

    if (tag === "select") {
      const opt = Array.from(target.options).find(function (o) { return o.value === String(value) || o.textContent.trim() === String(value); });
      target.value = opt ? opt.value : String(value);
    } else if (type === "checkbox" || type === "radio") {
      const shouldCheck = typeof value === "boolean" ? value : value === "true";
      if (target.checked !== shouldCheck) target.click();
      return { success: true, checked: target.checked };
    } else if (target.contentEditable === "true") {
      target.textContent = String(value);
    } else if (["input","textarea"].includes(tag)) {
      const proto = tag === "textarea" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, "value") ? Object.getOwnPropertyDescriptor(proto, "value").set : null;
      if (setter) setter.call(target, String(value));
      else target.value = String(value);
    } else {
      try { target.value = String(value); }
      catch (e) { return { error: "Cannot set value on <" + tag + "> element. No input found inside." }; }
    }

    target.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
    target.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
    return { success: true, value: target.value };
  }

  function getRefCoordinates(refId) {
    const el = resolveRef(refId);
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return { x: Math.round(rect.x + rect.width / 2), y: Math.round(rect.y + rect.height / 2) };
  }

  function scrollToRef(refId) {
    const el = resolveRef(refId);
    if (!el) return false;
    el.scrollIntoView({ block: "center", behavior: "instant" });
    return true;
  }

  window.__openagentChrome = {
    generateAccessibilityTree: generateAccessibilityTree,
    getPageText: getPageText,
    findElements: findElements,
    setFormValue: setFormValue,
    getRefCoordinates: getRefCoordinates,
    scrollToRef: scrollToRef,
    resolveRef: resolveRef,
    elementMap: elementMap,
  };
})();
`;
