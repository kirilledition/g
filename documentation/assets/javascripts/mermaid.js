(function () {
  var mermaidInitialized = false;

  function promoteMermaidCodeBlocks(root) {
    var sourceRoot = root || document;
    var candidates = [];

    if (
      sourceRoot instanceof Element &&
      sourceRoot.matches("pre.mermaid, pre > code.language-mermaid, pre > code.mermaid")
    ) {
      candidates.push(sourceRoot);
    }

    candidates.push.apply(
      candidates,
      Array.from(
        sourceRoot.querySelectorAll(
          "pre.mermaid, pre > code.language-mermaid, pre > code.mermaid",
        ),
      ),
    );

    for (var candidate of candidates) {
      var pre = candidate.matches("pre") ? candidate : candidate.closest("pre");
      if (!pre || pre.dataset.gMermaidPromoted === "true") {
        continue;
      }

      var code = pre.matches("pre") ? pre.querySelector("code") : candidate;
      var container = document.createElement("div");
      container.className = "mermaid";
      container.textContent = code ? code.textContent || "" : pre.textContent || "";
      pre.dataset.gMermaidPromoted = "true";
      pre.replaceWith(container);
    }
  }

  function initializeMermaid(root) {
    if (!window.mermaid) {
      return;
    }

    promoteMermaidCodeBlocks(root || document);

    if (!mermaidInitialized) {
      window.mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: "default",
      });
      mermaidInitialized = true;
    }

    var renderPromise = window.mermaid.run({
      querySelector: '.mermaid:not([data-processed="true"])',
    });

    if (renderPromise && typeof renderPromise.catch === "function") {
      renderPromise.catch(function (error) {
        console.error("Mermaid render failed", error);
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    initializeMermaid(document);
  });

  document.addEventListener("g:content-loaded", function (event) {
    initializeMermaid(event.detail && event.detail.root ? event.detail.root : document);
  });

  if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
      initializeMermaid(document);
    });
  }
})();
