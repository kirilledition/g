(function () {
  function initializeMermaid() {
    if (!window.mermaid) {
      return;
    }

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "default",
    });

    var renderPromise = window.mermaid.run({
      querySelector: ".mermaid",
    });

    if (renderPromise && typeof renderPromise.catch === "function") {
      renderPromise.catch(function (error) {
        console.error("Mermaid render failed", error);
      });
    }
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(initializeMermaid);
  } else {
    document.addEventListener("DOMContentLoaded", initializeMermaid);
  }
})();
