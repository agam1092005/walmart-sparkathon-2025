(function() {
  // Record the time the page script starts running
  const pageLoadTime = Date.now();
  let hasMovedMouse = false;
  let hasScrolled = false;
  let multipleTabs = false;
  const TAB_KEY = 'grinch-detector-tab-count';
  let tabId = Math.random().toString(36).substr(2, 9);

  // --- Heuristic 1: Listen for mouse movement ---
  document.addEventListener('mousemove', function() {
      hasMovedMouse = true;
  }, { once: true }); // We only need to know if it moved at all

  // --- Heuristic 2: Listen for scroll activity ---
  window.addEventListener('scroll', function() {
      hasScrolled = true;
  }, { once: true });

  // --- Heuristic 3: Detect multiple tabs for the same domain ---
  function updateTabCount(delta) {
      let count = parseInt(localStorage.getItem(TAB_KEY) || '0', 10);
      count += delta;
      localStorage.setItem(TAB_KEY, count);
      return count;
  }
  // On load, increment tab count
  let currentTabCount = updateTabCount(1);
  if (currentTabCount > 4) multipleTabs = true;
  // On unload, decrement tab count
  window.addEventListener('beforeunload', function() {
      updateTabCount(-1);
  });

  // Show popup if 4th tab is opened
  if (currentTabCount === 4) {
      showSuspiciousPopup(['Multiple Tabs Open for This Domain (4 tabs detected)']);
  }

  // --- Find and monitor potential 'Add to Cart' buttons ---
  const buttonKeywords = ['add to cart', 'buy now', 'add to bag', 'purchase'];
  const allButtons = document.querySelectorAll('button, a, input[type="submit"], input[type="button"]');

  allButtons.forEach(button => {
      const buttonText = (button.innerText || button.value || '').toLowerCase();
      const isTargetButton = buttonKeywords.some(keyword => buttonText.includes(keyword));
      if (isTargetButton) {
          button.addEventListener('click', function(event) {
              checkIfBot(event);
          });
      }
  });

  function checkIfBot(event) {
      const timeToAction = Date.now() - pageLoadTime;
      let suspiciousScore = 0;
      let reasons = [];

      // Check 1: Was the action incredibly fast?
      if (timeToAction < 1500) { 
          suspiciousScore++;
          reasons.push('Inhuman Speed (action taken in ' + timeToAction + 'ms)');
      }
      // Check 2: Was the mouse ever moved?
      if (!hasMovedMouse) {
          suspiciousScore++;
          reasons.push('No Mouse Movement Detected');
      }
      // Check 3: Was the page scrolled?
      if (!hasScrolled) {
          suspiciousScore++;
          reasons.push('No Scroll Activity Detected');
      }
      // Check 4: Multiple tabs open for this domain (more than 4)
      let tabCount = parseInt(localStorage.getItem(TAB_KEY) || '0', 10);
      if (tabCount > 4) {
          suspiciousScore++;
          reasons.push('Multiple Tabs Open for This Domain (' + tabCount + ' tabs detected)');
      }
      // If any of our checks failed, show the popup
      if (suspiciousScore > 0) {
          event.preventDefault();
          event.stopImmediatePropagation();
          // If the only reason is 'No Scroll Activity Detected', show yellow popup
          if (reasons.length === 1 && reasons[0] === 'No Scroll Activity Detected') {
              showSuspiciousPopup(reasons, 'yellow');
          } else {
              showSuspiciousPopup(reasons, 'red');
          }
      }
  }

  function showSuspiciousPopup(reasons, color) {
      if (document.getElementById('grinch-detector-popup')) return;
      const popup = document.createElement('div');
      popup.id = 'grinch-detector-popup';
      popup.style.position = 'fixed';
      popup.style.top = '20px';
      popup.style.right = '20px';
      popup.style.padding = '20px';
      popup.style.backgroundColor = color === 'yellow' ? '#FFD600' : '#c82424';
      popup.style.color = color === 'yellow' ? 'black' : 'white';
      popup.style.border = color === 'yellow' ? '2px solid #FFD600' : '2px solid #ff0000';
      popup.style.borderRadius = '10px';
      popup.style.zIndex = '99999';
      popup.style.fontFamily = 'Arial, sans-serif';
      popup.style.boxShadow = '0 4px 8px rgba(0,0,0,0.3)';
      popup.style.fontSize = '14px';
      let content = '<h3 style="margin: 0 0 10px 0; color: ' + (color === 'yellow' ? 'black' : 'white') + '; font-size: 16px;">Suspicious Activity Detected</h3>';
      content += '<ul style="margin: 0; padding-left: 20px;">';
      reasons.forEach(reason => {
          content += `<li>${reason}</li>`;
      });
      content += '</ul>';
      popup.innerHTML = content;
      document.body.appendChild(popup);
      setTimeout(() => {
          popup.remove();
      }, 6000);
  }

})();