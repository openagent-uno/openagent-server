// Cosmetic tab-group service worker for OpenAgent's dedicated browser.
//
// Its ONLY job is to keep the agent's tabs collected in a single labelled
// "OpenAgent" tab group so the user can see, at a glance, which tabs the agent
// is driving — the same affordance claude-in-chrome gives. It performs NO
// automation and holds NO messaging channel: all browser control happens over
// CDP, completely independent of this extension. If the extension fails to load
// (e.g. a browser that blocks --load-extension), automation is unaffected — you
// just don't get the visual group.

const GROUP_TITLE = "OpenAgent";
const GROUP_COLOR = "blue";

let pending = false;
function schedule() {
  if (pending) return;
  pending = true;
  // Debounce bursts of tab events into one regroup pass.
  setTimeout(() => { pending = false; ensureGrouped().catch(() => {}); }, 250);
}

async function ensureGrouped() {
  let windows;
  try {
    windows = await chrome.windows.getAll({ populate: true });
  } catch {
    return;
  }
  for (const w of windows) {
    if (w.type !== "normal" || !Array.isArray(w.tabs)) continue;
    const ungrouped = w.tabs
      .filter((t) => t.groupId === chrome.tabGroups.TAB_GROUP_ID_NONE && t.id >= 0)
      .map((t) => t.id);
    if (ungrouped.length === 0) continue;

    // Reuse an existing "OpenAgent" group in this window if present.
    let groupId;
    try {
      const existing = await chrome.tabGroups.query({ windowId: w.id, title: GROUP_TITLE });
      groupId = existing.length ? existing[0].id : undefined;
    } catch {
      groupId = undefined;
    }

    try {
      const opts = { tabIds: ungrouped };
      if (groupId !== undefined) opts.groupId = groupId;
      const gid = await chrome.tabs.group(opts);
      await chrome.tabGroups.update(gid, { title: GROUP_TITLE, color: GROUP_COLOR, collapsed: false });
    } catch {
      /* tabs may have closed mid-pass; next event re-runs */
    }
  }
}

chrome.runtime.onInstalled.addListener(schedule);
chrome.runtime.onStartup.addListener(schedule);
chrome.tabs.onCreated.addListener(schedule);
chrome.tabs.onAttached.addListener(schedule);
chrome.tabs.onUpdated.addListener((_id, info) => { if (info.status === "complete" || info.url) schedule(); });

schedule();
