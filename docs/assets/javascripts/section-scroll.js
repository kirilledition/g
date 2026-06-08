(() => {
  const articleSelector = "article.md-content__inner.md-typeset";
  const loadingClassName = "g-section-scroll-loading";
  const appendedPageClassName = "g-section-scroll-page";
  const loadAheadPixels = 900;

  const state = {
    exhausted: false,
    loading: false,
    nextPageUrl: getNextPageUrl(document, document.baseURI),
    scheduled: false,
    sectionTitle: getTopLevelSectionTitle(document),
    seenPageUrls: new Set([normalizePageUrl(document.location.href)]),
  };

  if (!state.nextPageUrl || !state.sectionTitle) {
    return;
  }

  window.addEventListener("scroll", scheduleLoadCheck, { passive: true });
  window.addEventListener("resize", scheduleLoadCheck);
  scheduleLoadCheck();

  function scheduleLoadCheck() {
    if (state.scheduled || state.exhausted || state.loading) {
      return;
    }

    state.scheduled = true;
    window.requestAnimationFrame(() => {
      state.scheduled = false;
      if (shouldLoadNextPage()) {
        void loadNextPage();
      }
    });
  }

  function shouldLoadNextPage() {
    const documentElement = document.documentElement;
    const remainingPixels =
      documentElement.scrollHeight - window.scrollY - window.innerHeight;
    return remainingPixels <= Math.max(loadAheadPixels, window.innerHeight);
  }

  async function loadNextPage() {
    if (state.loading || state.exhausted || !state.nextPageUrl) {
      return;
    }

    state.loading = true;
    document.documentElement.classList.add(loadingClassName);

    try {
      const pageUrl = state.nextPageUrl;
      const response = await window.fetch(pageUrl, { credentials: "same-origin" });
      if (!response.ok) {
        state.exhausted = true;
        return;
      }

      const fetchedDocument = new DOMParser().parseFromString(
        await response.text(),
        "text/html",
      );
      if (getTopLevelSectionTitle(fetchedDocument) !== state.sectionTitle) {
        state.exhausted = true;
        return;
      }

      const fetchedArticle = fetchedDocument.querySelector(articleSelector);
      if (!fetchedArticle) {
        state.exhausted = true;
        return;
      }

      appendPage(fetchedArticle, pageUrl, fetchedDocument.title);
      state.seenPageUrls.add(normalizePageUrl(pageUrl));

      const nextPageUrl = getNextPageUrl(fetchedDocument, pageUrl);
      if (nextPageUrl && !state.seenPageUrls.has(normalizePageUrl(nextPageUrl))) {
        state.nextPageUrl = nextPageUrl;
      } else {
        state.exhausted = true;
      }
    } catch {
      state.exhausted = true;
    } finally {
      state.loading = false;
      document.documentElement.classList.remove(loadingClassName);
      scheduleLoadCheck();
    }
  }

  function appendPage(fetchedArticle, pageUrl, pageTitle) {
    const currentArticle = document.querySelector(articleSelector);
    if (!currentArticle) {
      state.exhausted = true;
      return;
    }

    const appendedPage = document.createElement("section");
    appendedPage.className = appendedPageClassName;
    appendedPage.dataset.sectionScrollSource = pageUrl;
    appendedPage.setAttribute("aria-label", normalizeText(pageTitle));

    for (const child of Array.from(fetchedArticle.children)) {
      appendedPage.appendChild(child.cloneNode(true));
    }

    for (const action of appendedPage.querySelectorAll(".md-content__button")) {
      action.remove();
    }

    rewriteRelativeUrls(appendedPage, pageUrl);
    prefixLocalAnchors(appendedPage, pageUrl);
    currentArticle.appendChild(appendedPage);
  }

  function getNextPageUrl(sourceDocument, pageUrl) {
    const nextLink = sourceDocument.querySelector('link[rel="next"]');
    const href = nextLink?.getAttribute("href");
    return href ? new URL(href, pageUrl).href : null;
  }

  function getTopLevelSectionTitle(sourceDocument) {
    const navigationList = sourceDocument.querySelector(
      ".md-nav--primary > .md-nav__list",
    );
    if (!navigationList) {
      return "";
    }

    for (const item of Array.from(navigationList.children)) {
      if (
        !item.classList.contains("md-nav__item--active") ||
        !item.classList.contains("md-nav__item--section")
      ) {
        continue;
      }

      const label = Array.from(item.children).find((child) =>
        child.classList.contains("md-nav__link"),
      );
      return normalizeText(label?.textContent || "");
    }

    return "";
  }

  function rewriteRelativeUrls(container, pageUrl) {
    for (const element of container.querySelectorAll("[href], [src], [srcset]")) {
      rewriteUrlAttribute(element, "href", pageUrl);
      rewriteUrlAttribute(element, "src", pageUrl);
      rewriteSrcsetAttribute(element, pageUrl);
    }
  }

  function rewriteUrlAttribute(element, attributeName, pageUrl) {
    const value = element.getAttribute(attributeName);
    if (!value || !shouldRewriteUrl(value)) {
      return;
    }

    element.setAttribute(attributeName, new URL(value, pageUrl).href);
  }

  function rewriteSrcsetAttribute(element, pageUrl) {
    const value = element.getAttribute("srcset");
    if (!value) {
      return;
    }

    const rewrittenCandidates = value.split(",").map((candidate) => {
      const parts = candidate.trim().split(/\s+/);
      if (!parts[0] || !shouldRewriteUrl(parts[0])) {
        return candidate.trim();
      }

      parts[0] = new URL(parts[0], pageUrl).href;
      return parts.join(" ");
    });
    element.setAttribute("srcset", rewrittenCandidates.join(", "));
  }

  function shouldRewriteUrl(value) {
    const trimmedValue = value.trim();
    return (
      Boolean(trimmedValue) &&
      !trimmedValue.startsWith("#") &&
      !/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(trimmedValue)
    );
  }

  function prefixLocalAnchors(container, pageUrl) {
    const pageKey = buildPageKey(pageUrl);
    const idMap = new Map();

    for (const element of container.querySelectorAll("[id]")) {
      const originalId = element.id;
      const prefixedId = `${pageKey}-${originalId}`;
      idMap.set(originalId, prefixedId);
      element.id = prefixedId;
    }

    for (const link of container.querySelectorAll('a[href^="#"]')) {
      const originalId = decodeHashIdentifier(link.getAttribute("href").slice(1));
      const prefixedId = idMap.get(originalId);
      if (prefixedId) {
        link.setAttribute("href", `#${encodeURIComponent(prefixedId)}`);
      }
    }

    rewriteIdReferenceAttributes(container, idMap);
  }

  function rewriteIdReferenceAttributes(container, idMap) {
    const selector = [
      "[aria-controls]",
      "[aria-describedby]",
      "[aria-labelledby]",
      "[for]",
    ].join(",");

    for (const element of container.querySelectorAll(selector)) {
      rewriteIdReferenceAttribute(element, "aria-controls", idMap);
      rewriteIdReferenceAttribute(element, "aria-describedby", idMap);
      rewriteIdReferenceAttribute(element, "aria-labelledby", idMap);
      rewriteIdReferenceAttribute(element, "for", idMap);
    }
  }

  function rewriteIdReferenceAttribute(element, attributeName, idMap) {
    const value = element.getAttribute(attributeName);
    if (!value) {
      return;
    }

    const rewrittenValue = value
      .split(/\s+/)
      .map((identifier) => idMap.get(identifier) || identifier)
      .join(" ");
    element.setAttribute(attributeName, rewrittenValue);
  }

  function buildPageKey(pageUrl) {
    const url = new URL(pageUrl, document.baseURI);
    const pathKey = url.pathname
      .replace(/\/index\.html$/, "")
      .replace(/\/$/, "")
      .split("/")
      .filter(Boolean)
      .join("-");
    return (pathKey || "home").replace(/[^a-zA-Z0-9_-]/g, "-");
  }

  function decodeHashIdentifier(value) {
    try {
      return decodeURIComponent(value);
    } catch {
      return value;
    }
  }

  function normalizePageUrl(pageUrl) {
    const url = new URL(pageUrl, document.baseURI);
    url.hash = "";
    return url.href;
  }

  function normalizeText(value) {
    return value.replace(/\s+/g, " ").trim();
  }
})();
