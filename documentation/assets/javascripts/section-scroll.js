(() => {
  const articleSelector = "article.md-content__inner.md-typeset";
  const loadingClassName = "g-section-scroll-loading";
  const sectionPageClassName = "g-section-scroll-page";
  const currentPageClassName = "g-section-scroll-page--current";

  const currentArticle = document.querySelector(articleSelector);
  const sectionTitle = getTopLevelSectionTitle(document);
  if (!currentArticle || !sectionTitle) {
    return;
  }

  const state = {
    loading: false,
    pageHashIdentifierMaps: new Map(),
    pageTargets: new Map(),
    sectionTitle,
    seenPageUrls: new Set(),
  };

  const currentPageUrl = normalizePageUrl(document.location.href);
  const currentPageSection = wrapCurrentPage(currentArticle);
  state.seenPageUrls.add(currentPageUrl);
  state.pageTargets.set(currentPageUrl, getPageScrollTarget(currentPageSection));

  document.addEventListener("click", handleDocumentClick);
  window.addEventListener("popstate", scrollToCurrentHash);

  void loadSectionPages();

  async function loadSectionPages() {
    if (state.loading) {
      return;
    }

    state.loading = true;
    document.documentElement.classList.add(loadingClassName);

    try {
      await loadPageDirection(getPreviousPageUrl(document, document.baseURI), "previous");
      await loadPageDirection(getNextPageUrl(document, document.baseURI), "next");
      scrollToCurrentHash();
    } finally {
      state.loading = false;
      document.documentElement.classList.remove(loadingClassName);
    }
  }

  async function loadPageDirection(pageUrl, direction) {
    let candidatePageUrl = pageUrl;

    while (candidatePageUrl) {
      const normalizedPageUrl = normalizePageUrl(candidatePageUrl);
      if (state.seenPageUrls.has(normalizedPageUrl)) {
        return;
      }

      const loadedPage = await fetchSectionPage(candidatePageUrl);
      if (!loadedPage) {
        return;
      }

      insertPage(loadedPage.article, loadedPage.url, loadedPage.title, direction);
      state.seenPageUrls.add(normalizedPageUrl);
      candidatePageUrl =
        direction === "previous"
          ? getPreviousPageUrl(loadedPage.document, loadedPage.url)
          : getNextPageUrl(loadedPage.document, loadedPage.url);
    }
  }

  async function fetchSectionPage(pageUrl) {
    try {
      const response = await window.fetch(pageUrl, { credentials: "same-origin" });
      if (!response.ok) {
        return null;
      }

      const fetchedDocument = new DOMParser().parseFromString(
        await response.text(),
        "text/html",
      );
      if (getTopLevelSectionTitle(fetchedDocument) !== state.sectionTitle) {
        return null;
      }

      const fetchedArticle = fetchedDocument.querySelector(articleSelector);
      if (!fetchedArticle) {
        return null;
      }

      return {
        article: fetchedArticle,
        document: fetchedDocument,
        title: fetchedDocument.title,
        url: new URL(pageUrl, document.baseURI).href,
      };
    } catch {
      return null;
    }
  }

  function wrapCurrentPage(article) {
    const currentPage = document.createElement("section");
    currentPage.className = `${sectionPageClassName} ${currentPageClassName}`;
    currentPage.dataset.sectionScrollSource = normalizePageUrl(document.location.href);
    currentPage.setAttribute("aria-label", normalizeText(document.title));

    while (article.firstChild) {
      currentPage.appendChild(article.firstChild);
    }

    article.appendChild(currentPage);
    return currentPage;
  }

  function insertPage(fetchedArticle, pageUrl, pageTitle, direction) {
    const insertedPage = buildPageSection(fetchedArticle, pageUrl, pageTitle);

    if (direction === "previous") {
      const previousScrollHeight = document.documentElement.scrollHeight;
      currentArticle.insertBefore(insertedPage, currentArticle.firstChild);
      const scrollHeightDelta = document.documentElement.scrollHeight - previousScrollHeight;
      if (scrollHeightDelta) {
        window.scrollBy(0, scrollHeightDelta);
      }
      return;
    }

    currentArticle.appendChild(insertedPage);
  }

  function buildPageSection(fetchedArticle, pageUrl, pageTitle) {
    const sectionPage = document.createElement("section");
    sectionPage.className = sectionPageClassName;
    sectionPage.dataset.sectionScrollSource = pageUrl;
    sectionPage.setAttribute("aria-label", normalizeText(pageTitle));

    for (const child of Array.from(fetchedArticle.children)) {
      sectionPage.appendChild(child.cloneNode(true));
    }

    for (const action of sectionPage.querySelectorAll(".md-content__button")) {
      action.remove();
    }

    rewriteRelativeUrls(sectionPage, pageUrl);
    const hashIdentifierMap = prefixLocalAnchors(sectionPage, pageUrl);
    const normalizedPageUrl = normalizePageUrl(pageUrl);
    state.pageHashIdentifierMaps.set(normalizedPageUrl, hashIdentifierMap);
    state.pageTargets.set(normalizedPageUrl, getPageScrollTarget(sectionPage));
    return sectionPage;
  }

  function handleDocumentClick(event) {
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey
    ) {
      return;
    }

    const clickedElement = event.target instanceof Element ? event.target : null;
    const link = clickedElement?.closest("a[href]");
    if (!link || (link.target && link.target !== "_self") || link.hasAttribute("download")) {
      return;
    }

    const targetUrl = new URL(link.getAttribute("href"), document.baseURI);
    const targetElement = getLoadedPageTarget(targetUrl);
    if (!targetElement) {
      return;
    }

    event.preventDefault();
    updateHash(targetElement);
    targetElement.scrollIntoView({ block: "start" });
    closeNavigationDrawer();
  }

  function scrollToCurrentHash() {
    if (!window.location.hash) {
      return;
    }

    const targetElement = getLoadedPageTarget(new URL(window.location.href));
    if (targetElement) {
      targetElement.scrollIntoView({ block: "start" });
    }
  }

  function getLoadedPageTarget(targetUrl) {
    if (targetUrl.hash && isCurrentDocumentUrl(targetUrl)) {
      const currentHashTarget = document.getElementById(
        decodeHashIdentifier(targetUrl.hash.slice(1)),
      );
      if (currentHashTarget) {
        return currentHashTarget;
      }
    }

    const normalizedPageUrl = normalizePageUrl(targetUrl.href);
    if (!state.pageTargets.has(normalizedPageUrl)) {
      return null;
    }

    if (targetUrl.hash) {
      const originalIdentifier = decodeHashIdentifier(targetUrl.hash.slice(1));
      const hashIdentifierMap = state.pageHashIdentifierMaps.get(normalizedPageUrl);
      const targetIdentifier = hashIdentifierMap?.get(originalIdentifier) || originalIdentifier;
      const hashTarget = document.getElementById(targetIdentifier);
      if (hashTarget) {
        return hashTarget;
      }
    }

    return state.pageTargets.get(normalizedPageUrl) || null;
  }

  function isCurrentDocumentUrl(targetUrl) {
    return (
      targetUrl.origin === window.location.origin &&
      targetUrl.pathname === window.location.pathname &&
      targetUrl.search === window.location.search
    );
  }

  function updateHash(targetElement) {
    if (!targetElement.id) {
      return;
    }

    window.history.pushState(null, "", `#${encodeURIComponent(targetElement.id)}`);
  }

  function closeNavigationDrawer() {
    const drawerToggle = document.getElementById("__drawer");
    if (drawerToggle instanceof HTMLInputElement) {
      drawerToggle.checked = false;
    }
  }

  function getPageScrollTarget(sectionPage) {
    return sectionPage.querySelector("h1[id]") || sectionPage;
  }

  function getPreviousPageUrl(sourceDocument, pageUrl) {
    const previousLink = sourceDocument.querySelector('link[rel="prev"]');
    const href = previousLink?.getAttribute("href");
    return href ? new URL(href, pageUrl).href : null;
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
    return idMap;
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
