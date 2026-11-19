(() => {
  'use strict';
  document.documentElement.classList.add('js');
  const motionButton = document.querySelector('.motion-toggle');
  // Ambient motion starts playing; the visitor can pause it with the header control.
  let paused = false;
  let heroVisible = true;
  let animationId = 0;

  // Empty research links are intentional. Adding a URL makes them active.
  document.querySelectorAll('.pending-link').forEach(link => {
    if (link.getAttribute('href').trim()) {
      link.removeAttribute('aria-disabled');
      link.removeAttribute('aria-label');
      link.classList.remove('pending-link');
    }
    link.addEventListener('click', event => {
      if (!link.getAttribute('href').trim()) event.preventDefault();
    });
  });

  const header = document.querySelector('.site-header');
  const progress = document.querySelector('.reading-progress');
  let scrollQueued = false;
  const updateScroll = () => {
    const distance = document.documentElement.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${distance > 0 ? scrollY / distance : 0})`;
    header.classList.toggle('scrolled', scrollY > 20);
    scrollQueued = false;
  };
  addEventListener('scroll', () => {
    if (!scrollQueued) { scrollQueued = true; requestAnimationFrame(updateScroll); }
  }, { passive: true });
  updateScroll();

  const navigation = document.querySelector('#navigation');
  const menuButton = document.querySelector('.menu-toggle');
  function closeMenu() {
    menuButton.setAttribute('aria-expanded', 'false');
    menuButton.setAttribute('aria-label', 'Open navigation');
    navigation.classList.remove('open');
  }
  menuButton.addEventListener('click', () => {
    const open = menuButton.getAttribute('aria-expanded') !== 'true';
    menuButton.setAttribute('aria-expanded', String(open));
    menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    navigation.classList.toggle('open', open);
  });
  navigation.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
  document.addEventListener('click', event => { if (!header.contains(event.target)) closeMenu(); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') {
      closeMenu(); menuButton.focus();
    }
  });

  if ('IntersectionObserver' in window) {
    const revealObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) { entry.target.classList.add('visible'); revealObserver.unobserve(entry.target); }
      });
    }, { threshold: 0.08 });
    document.querySelectorAll('.reveal').forEach(element => revealObserver.observe(element));
    const sections = [...document.querySelectorAll('.snap-section')];
    const navObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) navigation.querySelectorAll('a').forEach(link => {
          const active = link.hash === `#${entry.target.id}`;
          link.classList.toggle('active', active);
          if (active) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      });
    }, { rootMargin: '-15% 0px -65% 0px' });
    sections.forEach(section => navObserver.observe(section));
  } else document.querySelectorAll('.reveal').forEach(element => element.classList.add('visible'));

  // Native tab behavior supports pointer, keyboard arrows, Home, and End.
  function setupTabs(list, activate) {
    const tabs = [...list.querySelectorAll('[role="tab"]')];
    function select(tab) {
      tabs.forEach(item => {
        item.setAttribute('aria-selected', String(item === tab));
        item.tabIndex = item === tab ? 0 : -1;
      });
      activate(tab);
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => select(tab));
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % tabs.length;
        else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index - 1 + tabs.length) % tabs.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = tabs.length - 1;
        if (next !== undefined) { event.preventDefault(); select(tabs[next]); tabs[next].focus(); }
      });
    });
  }
  const stages = {
    calibrate: { title: 'Let the policy’s outcomes lead.', body: 'Evaluate the real-only base policy on 4,000 held-out real scenes. Its driving outcomes define a policy-specific direction for valuing demonstrations.', key: '4,000 real scenes · Policy-specific feedback' },
    value: { title: 'Find the learning signal that matters.', body: 'Combine supervised trajectory gradients with regularized local geometry. A shared Gaussian projection makes influence-based valuation scalable across the candidate pool.', key: '2,048 projected dimensions · One linear solve' },
    curate: { title: 'Keep the value. Train the whole policy.', body: 'Rank demonstrations for the target planner, retain the top-valued subset, and combine it with real training data. Retrain the complete policy with its native objective.', key: 'Fixed data budget · Full-policy learning' }
  };
  const methodPanel = document.querySelector('#method-panel');
  function softenChange(element) {
    if (paused || !element.animate) return;
    element.getAnimations().forEach(animation => animation.cancel());
    element.animate([
      { opacity: .5, transform: 'translateY(7px)' },
      { opacity: 1, transform: 'translateY(0)' }
    ], { duration: 420, easing: 'cubic-bezier(.22,.8,.22,1)' });
  }
  setupTabs(document.querySelector('.method-tabs'), tab => {
    const stage = stages[tab.dataset.step];
    methodPanel.setAttribute('aria-labelledby', tab.id);
    methodPanel.querySelector('h3').textContent = stage.title;
    methodPanel.querySelector('p').textContent = stage.body;
    methodPanel.querySelector('.method-key').textContent = stage.key;
    softenChange(methodPanel);
  });

  // Paper-reported best NavTest budgets. NavHard values are independently best.
  const policies = {
    dd: { name: 'DiffusionDrive', real: 84.8, full: 85.9, pad: 88.3, retention: 25, hard: 32.9, hardGain: 5.0, caption: 'A quarter of the synthetic data.\nA stronger DiffusionDrive.' },
    ltf: { name: 'LTF', real: 79.1, full: 84.4, pad: 85.9, retention: 75, hard: 30.8, hardGain: 7.8, caption: 'Data selected for regression.\nA stronger LTF.' },
    mf: { name: 'MeanFuser', real: 84.2, full: 84.8, pad: 87.1, retention: 75, hard: 30.7, hardGain: 6.1, caption: 'Data selected for flow matching.\nA stronger MeanFuser.' }
  };
  const setText = (id, text) => { document.getElementById(id).textContent = text; };
  setupTabs(document.querySelector('.policy-tabs'), tab => {
    const data = policies[tab.dataset.policy];
    const gain = (data.pad - data.real).toFixed(1);
    document.querySelector('#result-panel').setAttribute('aria-labelledby', tab.id);
    setText('pad-score', data.pad.toFixed(1));
    setText('score-gain', `+${gain} points over real-only`);
    setText('result-caption', data.caption);
    setText('chart-title', `${data.name} on NavTest`);
    ['real', 'full', 'pad'].forEach(key => {
      setText(`${key}-value`, data[key].toFixed(1));
      document.getElementById(`${key}-bar`).style.setProperty('--value', `${data[key]}%`);
    });
    setText('retention-label', `${data.retention}% selected`);
    setText('retention-stat', `${data.retention}% retained`);
    document.querySelector('#budget-disc').style.setProperty('--retention', `${data.retention}%`);
    setText('hard-score', data.hard.toFixed(1));
    setText('hard-gain', `+${data.hardGain.toFixed(1)} vs. real-only`);
    softenChange(document.querySelector('.result-headline'));
    document.querySelector('.result-announcement').textContent = `${data.name}: PAD ${data.pad} EPDMS on NavTest, ${gain} points over real-only, retaining ${data.retention}% of synthetic data.`;
  });

  const dialog = document.querySelector('.image-dialog');
  let figureTrigger;
  document.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
    figureTrigger = button;
    const image = document.querySelector('#dialog-image');
    image.src = button.dataset.zoom;
    image.alt = button.querySelector('img').alt;
    setText('dialog-caption', button.dataset.caption);
    dialog.showModal();
    document.body.classList.add('dialog-open');
    document.querySelector('.dialog-image-wrap').scrollTo(0, 0);
  }));
  document.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => { document.body.classList.remove('dialog-open'); figureTrigger?.focus({ preventScroll: true }); });

  const dots = document.querySelector('#candidate-dots');
  for (let row = 0; row < 9; row++) for (let column = 0; column < 5; column++) {
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    const selected = (row * 7 + column * 3) % 5 === 0;
    Object.entries({ x: 24 + column * 31, y: 61 + row * 31, width: 13, height: 13, rx: 4, fill: selected ? '#6886c5' : ((row + column) % 3 === 0 ? '#e7b5c4' : '#becbe0') }).forEach(([key, value]) => dot.setAttribute(key, value));
    dot.classList.add('candidate');
    dot.style.setProperty('--delay', `${-(row + column * 1.7)}s`);
    dots.append(dot);
  }

  // Ambient flow: autonomous movement only; no pointer tracking or trails.
  const canvas = document.querySelector('#value-field');
  const ctx = canvas.getContext('2d');
  let width = 0, height = 0, phase = 0, lastFrame = 0;
  const routeCount = 26;
  function position(t, route) {
    const shift = route - routeCount / 2;
    const a = 1 - t;
    const p0 = { x: -.12 * width, y: height * .32 + shift * 17 };
    const p1 = { x: .2 * width, y: height * 1.12 + shift * 5 };
    const p2 = { x: .69 * width, y: height * 1.24 + shift * 5 };
    const p3 = { x: width * 1.12, y: height * .12 + shift * 20 };
    return { x: a*a*a*p0.x + 3*a*a*t*p1.x + 3*a*t*t*p2.x + t*t*t*p3.x,
      y: a*a*a*p0.y + 3*a*a*t*p1.y + 3*a*t*t*p2.y + t*t*t*p3.y + Math.sin(t * 7 + phase * .12 + route * .12) * 6 };
  }
  function drawField() {
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    for (let route = 0; route < routeCount; route++) {
      const rose = route % 5 === 0;
      ctx.beginPath();
      for (let step = 0; step <= 70; step++) {
        const point = position(step / 70, route);
        if (!step) ctx.moveTo(point.x, point.y); else ctx.lineTo(point.x, point.y);
      }
      ctx.strokeStyle = rose ? 'rgba(211,153,174,.13)' : 'rgba(134,157,199,.12)';
      ctx.lineWidth = rose ? 1.65 : 1.15;
      ctx.stroke();
      if (route % 2 !== 0) continue;
      const t = (phase * .031 + route * .073) % 1;
      const trail = .045;
      for (let segment = 0; segment < 12; segment++) {
        const ta = t - trail + segment / 12 * trail;
        if (ta < 0) continue;
        const start = position(ta, route), end = position(ta + trail / 12, route);
        ctx.beginPath(); ctx.moveTo(start.x, start.y); ctx.lineTo(end.x, end.y);
        ctx.strokeStyle = rose ? `rgba(207,145,165,${segment / 27})` : `rgba(117,148,197,${segment / 29})`;
        ctx.lineWidth = 2.3; ctx.stroke();
      }
      const point = position(t, route);
      ctx.beginPath(); ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = rose ? '#dfb2c1' : '#a2b9db'; ctx.fill();
    }
  }
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    width = rect.width; height = rect.height;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    ctx?.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawField(); updateScroll();
  }
  function frame(now) {
    if (paused || !heroVisible || document.hidden || !ctx) { animationId = 0; lastFrame = 0; return; }
    if (!lastFrame || now - lastFrame >= 30) {
      phase += lastFrame ? Math.min((now - lastFrame) / 1000, .08) : 0;
      lastFrame = now; drawField();
    }
    animationId = requestAnimationFrame(frame);
  }
  function resumeField() { if (!animationId && !paused && heroVisible && !document.hidden) animationId = requestAnimationFrame(frame); }
  function setMotion(value) {
    paused = value;
    document.documentElement.classList.toggle('motion-paused', paused);
    motionButton.setAttribute('aria-pressed', String(paused));
    const label = paused ? 'Play ambient animation' : 'Pause ambient animation';
    motionButton.setAttribute('aria-label', label); motionButton.title = label;
    if (paused) {
      cancelAnimationFrame(animationId); animationId = 0; lastFrame = 0;
      [methodPanel, document.querySelector('.result-headline')].forEach(element =>
        element.getAnimations().forEach(animation => animation.cancel()));
    }
    else resumeField();
  }
  motionButton.addEventListener('click', () => setMotion(!paused));
  document.addEventListener('visibilitychange', () => { if (!document.hidden) resumeField(); });
  if ('IntersectionObserver' in window) new IntersectionObserver(entries => {
    heroVisible = entries[0].isIntersecting; resumeField();
  }).observe(document.querySelector('.hero'));
  if ('ResizeObserver' in window) new ResizeObserver(resizeCanvas).observe(canvas);
  else addEventListener('resize', resizeCanvas, { passive: true });
  resizeCanvas(); setMotion(paused);
})();
