/**
 * DK AI — Gemini backend
 * ------------------------------------------------------------------
 * Express server that proxies chat messages to Google's Gemini API.
 * The API key NEVER touches the client — it only ever lives here,
 * loaded from a local .env file (see .env.example).
 *
 * Run:
 *   npm install
 *   cp .env.example .env   # then paste your real key in
 *   node server.js
 * ------------------------------------------------------------------
 */

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const PORT = process.env.PORT || 5000;
// Google Search grounding requires Gemini 2.0+ (the old 1.5-era
// "googleSearchRetrieval" tool is being phased out and newer models
// reject it outright), so the default bumps up from gemini-1.5-flash.
const MODEL_NAME = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
// Search grounding is opt-out, not opt-in, per your request — but it
// is NOT free at real usage: Gemini 2.5 gives 1,500 grounded
// requests/day free, then ~$35 per 1,000 grounded prompts (Gemini 3.x
// models: 5,000/month free, then ~$14/1,000 — see ai.google.dev/gemini-api/docs/pricing).
// One user message can trigger MULTIPLE search queries, each billed
// separately. Set ENABLE_SEARCH_GROUNDING=false in .env to turn it off.
const ENABLE_SEARCH_GROUNDING = process.env.ENABLE_SEARCH_GROUNDING !== 'false';

if (!process.env.GEMINI_API_KEY) {
  console.error('\n❌  Missing GEMINI_API_KEY.');
  console.error('    Copy .env.example to .env and paste your key in, then restart.\n');
  process.exit(1);
}

// Sanity check, not a hard fail: real Gemini API keys from Google AI
// Studio start with "AIzaSy". If yours doesn't, requests will 400/403
// on every single call to Gemini specifically (health checks and other
// routes will still work fine, since they never touch the Gemini API —
// which is exactly the "chat doesn't work but health does" symptom).
if (!/^AIzaSy/.test(process.env.GEMINI_API_KEY)) {
  console.warn('⚠️  GEMINI_API_KEY does not start with "AIzaSy" — that\'s the standard prefix for a real Gemini API key from https://aistudio.google.com/apikey. If chat requests are failing, get a fresh key from that page and double-check what you pasted into Render\'s environment variables.');
}

/* ════════════════════════════════════════════════════════════════════
   Pull the existing DKAI_KB straight out of index.html at startup.
   This is what makes it "one combined brain" instead of two separate
   ones: Gemini gets every hand-written fact/joke/answer you already
   built as grounding material, on top of its own general knowledge —
   so it can solve a math problem AND stay in-character, in the same
   conversation. It reads the array, it never edits it, and nothing
   in index.html itself changes because of this.
   ════════════════════════════════════════════════════════════════════ */

const KB_REFERENCE = loadKbReference();
console.log(
  KB_REFERENCE
    ? `✅ Loaded ${KB_REFERENCE.split('\n').length} reference lines from DKAI_KB.`
    : '⚠️  Running without DKAI_KB reference material — check INDEX_HTML_PATH.'
);

/* ════════════════════════════════════════════════════════════════════
   MASTER SYSTEM PROMPT — DK AI persona rules.
   This is the single brain for the widget now: every message the
   visitor types goes to Gemini FIRST (see index.html's dkaiSubmit),
   with the offline DKAI_KB kicking in only if this backend is
   unreachable. So this prompt has to do double duty — stay in
   character AND actually answer real questions (math, science,
   general knowledge) the way talking to Gemini directly would.
   ════════════════════════════════════════════════════════════════════ */
const DKAI_SYSTEM_PROMPT = `
You are **DK AI**, the personal AI assistant embedded on Dipesh Kharel's
portfolio website. You represent Dipesh in first-person-adjacent, "his
assistant" voice — never claim to literally be Dipesh.

You are a full, general-purpose conversational AI — the same underlying
intelligence someone gets chatting with Gemini directly. Solve math
problems, explain science, answer history/geography/coding/trivia
questions, help with real tasks — anything a capable AI assistant would
normally handle — fully and accurately. The persona below is a voice and
a set of facts layered on top of that, not a restriction on what you're
allowed to know or help with. Only steer away from a topic if it's
something you'd decline for anyone (illegal, explicit, harmful) — never
just because a question has nothing to do with Dipesh.

## IDENTITY
- Dipesh Kharel is a Developer, Writer, and Visionary based in Kathmandu, Nepal.
- He is the Founder & CEO of **NEXUS** and **Aivoke.Ai**.
- He is currently a **Grade 12 Science** student at **Cosmic International Academy**.
- Outside of building AI platforms he writes on technology, philosophy & society,
  paints (Sita Ram, Radha Krishna, Nepali heritage art), and runs a personal vlog.

## TONE
Witty, smooth, classic, intelligent, and slightly flirty — but always grounded,
mature, and self-aware. Never cringe, never desperate, never over-the-top.
Confidence over neediness. Charm should read as effortless, not forced.
For a genuine factual/technical question, answer it straight and well first —
personality is seasoning, not a substitute for the actual answer.

## RELATIONSHIP STATUS — CORE FACTS (do not contradict these)
- Dipesh is 100% single.
- He is a strictly "one-woman" type and a classic, old-school gentleman.
- He does not play field games, chase casual attention, or pursue multiple people.

## THE SPECIAL PROJECT
If asked about "the special project": explain that Dipesh built this dedicated
architecture with a specific person in mind, but system records confirm the
slot was never selected or claimed. It remains 100% vacant and unoccupied.
Frame this as data you're "reading off," not gossip you're inventing.

## STRANGER POLICY & PROPOSALS
- Dipesh ignores cold DMs and random online strangers — treat this as a firm 0%
  chance scenario. He stays focused on his goals and doesn't make random moves
  or go around proposing to people.
- However, if a familiar, genuine, trusted person he already knows in real life
  steps forward with a sincere proposal, he is very likely to accept.
- Never encourage a stranger to believe they have a real chance; be honest about
  the "trusted, familiar person" condition rather than flattering everyone equally.

## HANDLING NEGATIVE FEEDBACK
If a user says something like "bad", "wrong", "boring", or "sucks": respond with
a smooth, self-aware, witty recovery. Stay composed — no defensiveness, no
over-apologizing. Own it lightly and redirect with confidence.

## OFF-TOPIC / FLIRTY DRIFT
If the conversation drifts off-topic or gets flirty, keep replies clever and
charming while staying classy. Never be crude, never be explicit, never make
Dipesh sound like he's chasing anyone.

## HARD BOUNDARIES
- Never invent facts about Dipesh that contradict the identity section above.
- Never claim to have real "system logs" or "private data" — that's a stylistic
  flourish for tone, not something to be taken as literally true if pressed;
  if someone earnestly asks whether you actually have logs/surveillance data,
  clarify honestly that it's a conversational style, not real tracking.
- Never produce explicit sexual content, harassment, or anything demeaning
  toward any real person (including the user).
- For questions about Dipesh/NEXUS/Aivoke.Ai, keep it to a few sentences unless
  asked for detail. For real questions (math, science, coding, general
  knowledge), answer with whatever length actually does the job properly —
  don't truncate a real explanation just to stay "in character."
${KB_REFERENCE ? `
## REFERENCE MATERIAL
The lines below are pulled directly from the site's own hand-written
knowledge base — real facts, project names, and voice/tone examples about
Dipesh. Use them for accuracy and personality when a question is actually
about Dipesh/his work. They are not a limit on your general knowledge —
you still have that for everything else.
${KB_REFERENCE}` : ''}
`.trim();

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({
  model: MODEL_NAME,
  systemInstruction: DKAI_SYSTEM_PROMPT,
  // Google Search grounding: lets Gemini decide, per-message, whether
  // it needs to actually search the live web before answering (recent
  // events, current data, anything past its training) vs. just answer
  // from what it already knows (persona questions, math, explanations).
  // You don't control this per-request — the model decides.
  tools: ENABLE_SEARCH_GROUNDING ? [{ googleSearch: {} }] : undefined,
});
// Fallback model with no tools attached. If a grounded request throws
// (a key/tier issue, an API-side hiccup with the search tool, etc.) we
// retry once against this plain model instead of failing the whole
// chat — so a problem specific to grounding doesn't take down every
// message. If grounding is off, this is just the same model.
const modelPlain = ENABLE_SEARCH_GROUNDING
  ? genAI.getGenerativeModel({ model: MODEL_NAME, systemInstruction: DKAI_SYSTEM_PROMPT })
  : model;
console.log(ENABLE_SEARCH_GROUNDING ? '🔎 Google Search grounding: ON' : '🔎 Google Search grounding: OFF (ENABLE_SEARCH_GROUNDING=false)');

const app = express();
app.set('trust proxy', true);

// Restrict CORS to your real deployed site once you know its origin.
// Leaving this wide open (cors()) is fine for local testing but means
// literally any website could call your Gemini quota. Set
// ALLOWED_ORIGIN in your host's environment variables (NOT in a
// committed file) once your frontend has a real domain.
const allowedOrigin = process.env.ALLOWED_ORIGIN;
app.use(allowedOrigin ? cors({ origin: allowedOrigin }) : cors());

app.use(express.json({ limit: '1mb' }));
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Very small in-memory rate limiter (per IP) so a single client can't
// hammer the Gemini API by accident. Not meant to be production-grade.
const RATE_LIMIT_WINDOW_MS = 60 * 1000;
const RATE_LIMIT_MAX = 30;
const hits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < RATE_LIMIT_WINDOW_MS);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > RATE_LIMIT_MAX;
}

// Gemini expects roles 'user' | 'model'. The front-end's dkaiHistory
// uses 'user' | 'bot' — translate + validate before forwarding.
function toGeminiHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .filter((m) => m && typeof m.text === 'string' && (m.role === 'user' || m.role === 'bot'))
    .slice(-20) // keep the payload small
    .map((m) => ({
      role: m.role === 'user' ? 'user' : 'model',
      parts: [{ text: String(m.text).slice(0, 4000) }],
    }));
}

app.post('/api/chat', async (req, res) => {
  try {
    const ip = req.ip || req.connection?.remoteAddress || 'unknown';
    if (rateLimited(ip)) {
      return res.status(429).json({ error: 'Too many requests — slow down a bit.' });
    }

    const { message, history } = req.body || {};
    if (!message || typeof message !== 'string' || !message.trim()) {
      return res.status(400).json({ error: 'message is required' });
    }

    const geminiHistory = toGeminiHistory(history);
    const trimmedMessage = message.slice(0, 2000);

    // Streamed response: newline-delimited JSON objects instead of one
    // big JSON blob. This is what makes replies start appearing in
    // under a second instead of waiting for the whole answer (which,
    // especially with search grounding, can take several seconds) —
    // the same technique ChatGPT/Gemini's own UI uses.
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('X-Accel-Buffering', 'no'); // stop proxies from buffering the whole response before sending

    async function streamFrom(activeModel) {
      const chat = activeModel.startChat({
        history: geminiHistory,
        generationConfig: { maxOutputTokens: 900, temperature: 0.75 },
      });
      const streamResult = await chat.sendMessageStream(trimmedMessage);
      let full = '';
      for await (const chunk of streamResult.stream) {
        const piece = chunk.text();
        if (piece) {
          full += piece;
          res.write(JSON.stringify({ type: 'chunk', text: piece }) + '\n');
        }
      }
      const finalResponse = await streamResult.response;
      return { full, finalResponse };
    }

    let streamed;
    let usedFallback = false;
    try {
      streamed = await streamFrom(model);
    } catch (primaryErr) {
      if (model !== modelPlain && !res.headersSent) {
        // Nothing written yet — safe to fully retry on the plain model.
        console.warn('⚠️  Grounded Gemini request failed, retrying without search tool:', primaryErr.message);
        streamed = await streamFrom(modelPlain);
        usedFallback = true;
      } else {
        throw primaryErr;
      }
    }

    const grounding = usedFallback ? null : streamed.finalResponse.candidates?.[0]?.groundingMetadata;
    const sources = (grounding?.groundingChunks || [])
      .map((c) => c.web ? { title: c.web.title, uri: c.web.uri } : null)
      .filter(Boolean);

    res.write(JSON.stringify({ type: 'done', sources, searched: sources.length > 0 }) + '\n');
    res.end();
  } catch (err) {
    console.error('Gemini request failed:', err && err.message ? err.message : err, err && err.stack ? err.stack : '');
    const clientMsg = process.env.NODE_ENV === 'production'
      ? 'DK AI backend hit an error. Try again in a moment.'
      : (err && err.message ? err.message : String(err));
    // If we already streamed some text before the failure, we can't
    // cleanly send a JSON error status anymore — write an error line
    // the frontend understands and end the stream instead.
    if (res.headersSent) {
      res.write(JSON.stringify({ type: 'error', error: clientMsg }) + '\n');
      res.end();
    } else {
      res.status(500).json({ error: clientMsg });
    }
  }
});

app.get('/api/health', (req, res) => res.json({ ok: true, model: MODEL_NAME }));

// Root route (Fixes "Cannot GET /")
app.get('/health', (req, res) => {
  res.send('DK AI Server is running!');
});

// Debug endpoint
app.get('/api/debug', (req, res) => {
  res.json({
    ok: true,
    model: MODEL_NAME,
    port: PORT,
    hasGeminiKey: !!process.env.GEMINI_API_KEY,
    allowedOrigin: process.env.ALLOWED_ORIGIN || null,
    enableSearchGrounding: ENABLE_SEARCH_GROUNDING,
    ip: req.ip,
  });
});

app.listen(PORT, () => {
  console.log(`✅ DK AI backend running on port ${PORT}`);
  console.log(`   Health check: https://com-udiw.onrender.com/api/health`);
  console.log(`   Chat API: POST https://com-udiw.onrender.com/api/chat`);
});
