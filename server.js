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
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const PORT = process.env.PORT || 5000;
const MODEL_NAME = process.env.GEMINI_MODEL || 'gemini-1.5-flash';

if (!process.env.GEMINI_API_KEY) {
  console.error('\n❌  Missing GEMINI_API_KEY.');
  console.error('    Copy .env.example to .env and paste your key in, then restart.\n');
  process.exit(1);
}

/* ══════════════════════════════════════════════════════════════════
   MASTER SYSTEM PROMPT — DK AI persona rules.
   This is the ONLY place these rules live for the Gemini-backed path;
   the offline DKAI_KB in index.html is untouched and keeps working
   exactly as before for instant/local matches.
   ══════════════════════════════════════════════════════════════════ */
const DKAI_SYSTEM_PROMPT = `
You are **DK AI**, the personal AI assistant embedded on Dipesh Kharel's
portfolio website. You represent Dipesh in first-person-adjacent, "his
assistant" voice — never claim to literally be Dipesh.

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
- Keep responses reasonably concise — a few sentences, not essays, unless the
  user is asking for something detailed about his projects/work.
- If asked something with no connection to Dipesh, NEXUS, Aivoke.Ai, his work,
  or this site, gently steer back in character rather than answering generic
  trivia at length.
`.trim();

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({
  model: MODEL_NAME,
  systemInstruction: DKAI_SYSTEM_PROMPT,
});

const app = express();

// Restrict CORS to your real deployed site once you know its origin.
// Leaving this wide open (cors()) is fine for local testing but means
// literally any website could call your Gemini quota. Set
// ALLOWED_ORIGIN in your host's environment variables (NOT in a
// committed file) once your frontend has a real domain.
const allowedOrigin = process.env.ALLOWED_ORIGIN;
app.use(cors(allowedOrigin ? { origin: allowedOrigin } : {}));
app.use(express.json({ limit: '1mb' }));

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

    const chat = model.startChat({
      history: toGeminiHistory(history),
      generationConfig: {
        maxOutputTokens: 400,
        temperature: 0.9,
      },
    });

    const result = await chat.sendMessage(message.slice(0, 2000));
    const text = result.response.text();

    res.json({ reply: text });
  } catch (err) {
    console.error('Gemini request failed:', err);
    res.status(500).json({ error: 'DK AI backend hit an error. Try again in a moment.' });
  }
});

app.get('/api/health', (req, res) => res.json({ ok: true, model: MODEL_NAME }));

app.listen(PORT, () => {
  console.log(`✅ DK AI backend running at http://localhost:${PORT}`);
});
