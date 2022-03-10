window.onload = () => {
    'use strict';

    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register("../static/sw.js").then(function() { console.log("Service Worker Registered"); });;
    }

  }