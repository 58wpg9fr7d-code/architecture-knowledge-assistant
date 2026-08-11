// ArchMind Chrome Extension — Background Service Worker
// Handles the right-click context menu and stores selected text.

const MENU_ID = "archmind-ask";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: "🔍 用 ArchMind 检索这段文字",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === MENU_ID && info.selectionText) {
    const text = info.selectionText.trim();
    if (text) {
      chrome.storage.local.set({ selectedText: text, timestamp: Date.now() }, () => {
        // Notify the user via a brief badge
        chrome.action.setBadgeText({ text: "●" });
        chrome.action.setBadgeBackgroundColor({ color: "#2563EB" });
        // Clear the badge after a few seconds
        setTimeout(() => {
          chrome.action.setBadgeText({ text: "" });
        }, 3000);
      });
    }
  }
});
