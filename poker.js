/* Poker Advisor — browser port of the Flask app's logic.
 * Faithful reimplementation, validated to match XGBoost predict_proba (~1e-7)
 * and the treys hand evaluator (exact) in the accompanying Node test.
 * Uses globals from data.js: window.POKER_MODELS, window.POKER_TREYS.
 */
(function (root) {
  const MODELS = root.POKER_MODELS;
  const TREYS = root.POKER_TREYS;
  const MAX = TREYS.MAX; // 7462
  const PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];
  const DECK = Object.values(TREYS.card_ints);

  // ── card helpers (treys integer encoding) ──
  function normalizeKey(s) {
    if (!s) return null;
    s = String(s).trim();
    if (!s) return null;
    if (/^10[shdc]$/i.test(s)) s = 'T' + s[2];
    const key = s[0].toUpperCase() + (s[1] ? s[1].toLowerCase() : '');
    return TREYS.card_ints[key] !== undefined ? key : null;
  }
  function toInt(key) { return TREYS.card_ints[key]; }
  const cardRank = (ci) => (ci >> 8) & 0xf;
  const cardSuit = (ci) => (ci >> 12) & 0xf;

  // ── hand evaluator (treys parity) ──
  function primeProductFromHand(cards) {
    let p = 1;
    for (const c of cards) p *= c & 0xff;
    return p;
  }
  function primeProductFromRankbits(rb) {
    let p = 1;
    for (let i = 0; i < 13; i++) if (rb & (1 << i)) p *= PRIMES[i];
    return p;
  }
  function five(c) {
    if ((c[0] & c[1] & c[2] & c[3] & c[4] & 0xf000) !== 0) {
      const handOR = (c[0] | c[1] | c[2] | c[3] | c[4]) >>> 16;
      return TREYS.flush_lookup[String(primeProductFromRankbits(handOR))];
    }
    return TREYS.unsuited_lookup[String(primeProductFromHand(c))];
  }
  function combos5(a) {
    const n = a.length, r = [];
    for (let i = 0; i < n - 4; i++)
      for (let j = i + 1; j < n - 3; j++)
        for (let k = j + 1; k < n - 2; k++)
          for (let l = k + 1; l < n - 1; l++)
            for (let m = l + 1; m < n; m++) r.push([a[i], a[j], a[k], a[l], a[m]]);
    return r;
  }
  function evaluateBest(cards) {
    if (cards.length === 5) return five(cards);
    let best = 1e9;
    for (const c of combos5(cards)) {
      const v = five(c);
      if (v < best) best = v;
    }
    return best;
  }

  // ── Monte Carlo equity ──
  function deckExcluding(known) {
    const set = new Set(known);
    const d = [];
    for (const c of DECK) if (!set.has(c)) d.push(c);
    for (let i = d.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      const t = d[i]; d[i] = d[j]; d[j] = t;
    }
    return d;
  }
  function preflopEquity(hole, trials = 500) {
    let wins = 0;
    for (let t = 0; t < trials; t++) {
      const d = deckExcluding(hole);
      const opp = [d[0], d[1]];
      const board = [d[2], d[3], d[4], d[5], d[6]];
      if (evaluateBest(board.concat(hole)) < evaluateBest(board.concat(opp))) wins++;
    }
    return wins / trials;
  }
  function winEquity(hole, boardKnown, trials = 500) {
    let wins = 0;
    const need = 5 - boardKnown.length;
    for (let t = 0; t < trials; t++) {
      const d = deckExcluding(hole.concat(boardKnown));
      const opp = [d[0], d[1]];
      const board = boardKnown.concat(d.slice(2, 2 + need));
      if (evaluateBest(board.concat(hole)) < evaluateBest(board.concat(opp))) wins++;
    }
    return Math.round((wins / trials) * 1000) / 10;
  }

  // ── XGBoost (multi:softprob) ──
  function evalTree(node, f) {
    while (node.leaf === undefined) {
      const v = f[node.f];
      if (v === undefined) node = node.miss === 'yes' ? node.yes : node.no;
      else node = v < node.t ? node.yes : node.no;
    }
    return node.leaf;
  }
  function predict(street, f) {
    const m = MODELS.models[street];
    const K = MODELS.num_class;
    const margins = m.base_score.slice();
    for (let i = 0; i < m.trees.length; i++) margins[i % K] += evalTree(m.trees[i], f);
    const mx = Math.max.apply(null, margins);
    const ex = margins.map((x) => Math.exp(x - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    const probs = ex.map((x) => x / sum);
    let bi = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bi]) bi = i;
    return { action: MODELS.classes[bi], confidence: probs[bi] * 100 };
  }

  function buildFeats(hole, street, flop, turn, river, strength) {
    const f = {
      hole1_rank: cardRank(hole[0]), hole1_suit: cardSuit(hole[0]),
      hole2_rank: cardRank(hole[1]), hole2_suit: cardSuit(hole[1]),
    };
    if (street !== 'preflop') {
      f.flop1_rank = cardRank(flop[0]); f.flop1_suit = cardSuit(flop[0]);
      f.flop2_rank = cardRank(flop[1]); f.flop2_suit = cardSuit(flop[1]);
      f.flop3_rank = cardRank(flop[2]); f.flop3_suit = cardSuit(flop[2]);
    }
    if (street === 'turn' || street === 'river') {
      f.turn_rank = cardRank(turn); f.turn_suit = cardSuit(turn);
    }
    if (street === 'river') {
      f.river_rank = cardRank(river); f.river_suit = cardSuit(river);
    }
    f.strength = strength;
    return f;
  }

  const strengthOf = (cards) => 1 - evaluateBest(cards) / MAX;

  // Full advice pipeline mirroring app.py's /predict route.
  function advise(input) {
    const hole = input.hole.map(toInt);
    const results = {};

    const eq = preflopEquity(hole);
    const pf = predict('preflop', buildFeats(hole, 'preflop', null, null, null, eq));
    results['pre-flop'] = { action: pf.action, confidence: pf.confidence, win_equity: Math.round(eq * 1000) / 10 };

    if (input.flop && input.flop.length === 3 && input.flop.every(Boolean)) {
      const flop = input.flop.map(toInt);
      const s = strengthOf(flop.concat(hole));
      const r = predict('flop', buildFeats(hole, 'flop', flop, null, null, s));
      results.flop = { action: r.action, confidence: r.confidence, win_equity: winEquity(hole, flop) };

      if (input.turn) {
        const turn = toInt(input.turn);
        const s2 = strengthOf(flop.concat([turn], hole));
        const r2 = predict('turn', buildFeats(hole, 'turn', flop, turn, null, s2));
        results.turn = { action: r2.action, confidence: r2.confidence, win_equity: winEquity(hole, flop.concat([turn])) };

        if (input.river) {
          const river = toInt(input.river);
          const s3 = strengthOf(flop.concat([turn, river], hole));
          const r3 = predict('river', buildFeats(hole, 'river', flop, turn, river, s3));
          results.river = { action: r3.action, confidence: r3.confidence, win_equity: winEquity(hole, flop.concat([turn, river])) };
        }
      }
    }
    return results;
  }

  root.PokerCore = {
    normalizeKey, toInt, evaluateBest, strengthOf, preflopEquity, winEquity,
    predict, buildFeats, advise, MAX,
  };
})(typeof window !== 'undefined' ? window : globalThis);
