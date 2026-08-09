/**
 * DK AI — Gemini backend (streaming)
 *
 * Express server that proxies chat messages to Google's Gemini API using
 * the current @google/genai SDK, streaming real tokens back to the
 * browser as newline-delimited JSON:
 *
 *   {"type":"chunk","text":"..."}   — one per streamed token/piece
 *   {"type":"done","sources":[...],"searched":bool}   — final line
 *   {"type":"error","error":"..."}  — only if something breaks mid-stream
 *
 * This exact shape matches what the existing index.html frontend already
 * parses in dkaiCallGeminiStreaming() (reads res.body via a ReadableStream
 * reader, splits on '\n', JSON.parses each line, switches on msg.type).
 * That parsing logic is UNCHANGED — this file only needs to emit lines
 * in the shape it already expects.
 *
 * Gemini is the ONLY provider used for /api/chat. OPENAI_API_KEY is read
 * only for the /api/debug report and is never used to generate a reply,
 * and there is no automatic fallback to OpenAI.
 */

require('dotenv').config();

const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');

const PORT = process.env.PORT || 5000;

const MODEL_NAME =
  process.env.GEMINI_MODEL || 'gemini-3.6-flash';

const ENABLE_SEARCH_GROUNDING = false;


/* ============================================================
   API KEYS
   ============================================================ */

if (!process.env.GEMINI_API_KEY) {
  console.error('\n❌ Missing GEMINI_API_KEY.');
  console.error(
    'Set GEMINI_API_KEY in the Render service environment variables.\n'
  );
  process.exit(1);
}

/*
 * Newer Google AI Studio keys can begin with "AQ." instead of the older
 * "AIzaSy" prefix. Both are valid. We do NOT validate the key's shape —
 * the Gemini API itself is the source of truth on whether it's accepted.
 * The key is read only from process.env.GEMINI_API_KEY, never logged,
 * hardcoded, or sent to the browser.
 *
 * OPENAI_API_KEY stays in the environment for future use only — it is
 * never used to generate a chat response in this file.
 */

function redactSecrets(input) {
  let text = String(input);

  const secrets = [
    process.env.GEMINI_API_KEY,
    process.env.OPENAI_API_KEY
  ].filter(Boolean);

  for (const secret of secrets) {
    if (secret.length >= 6) {
      text = text.split(secret).join('[REDACTED]');
    }
  }

  return text;
}


/* ============================================================
   LOAD DKAI_KB FROM INDEX.HTML (optional — safe if missing)
   ============================================================ */

function loadKbReference() {
  const indexPath =
    process.env.INDEX_HTML_PATH ||
    path.join(__dirname, 'index.html');

  try {
    const html = fs.readFileSync(indexPath, 'utf8');

    const scripts = [
      ...html.matchAll(/<script>([\s\S]*?)<\/script>/g)
    ].map((m) => m[1]);

    for (const script of scripts) {
      const startIdx = script.indexOf('var DKAI_KB = [');

      if (startIdx === -1) continue;

      const endIdx = script.indexOf('];', startIdx);

      if (endIdx === -1) continue;

      const arrCode = script.slice(startIdx, endIdx + 2);

      const sandbox = {};

      // Trusted self-authored index.html content.
      new Function(
        'sandbox',
        arrCode + '\nsandbox.DKAI_KB = DKAI_KB;'
      )(sandbox);

      const kb = sandbox.DKAI_KB;

      if (Array.isArray(kb) && kb.length) {
        return kb
          .map(
            (e) =>
              '- ' +
              String(e.answer || '')
                .replace(/<[^>]+>/g, '')
                .replace(/\*\*/g, '')
                .replace(/\s+/g, ' ')
                .trim()
          )
          .join('\n');
      }
    }
  } catch (err) {
    console.warn(
      `⚠️ Couldn't load DKAI_KB from ${indexPath}: ${err.message}`
    );
  }

  return '';
}

const KB_REFERENCE = loadKbReference();

console.log(
  KB_REFERENCE
    ? `✅ Loaded ${KB_REFERENCE.split('\n').length} DKAI reference lines.`
    : '⚠️ Running without DKAI_KB reference material.'
);


/* ============================================================
   DK AI SYSTEM PROMPT
   ============================================================ */

const DKAI_SYSTEM_PROMPT = `
You are DK AI, the personal AI assistant embedded on
Dipesh Kharel's portfolio website.

You are a FULL GENERAL-PURPOSE AI assistant.

You can answer:
- general knowledge
- science
- mathematics
- physics
- chemistry
- biology
- history
- geography
- programming
- technology
- coding
- explanations
- everyday questions

IMPORTANT:

A question DOES NOT need to be related to Dipesh Kharel,
NEXUS, or Aivoke.Ai.

If the user asks a normal general question, ANSWER IT DIRECTLY.

For example:

User: "What is a computer?"
Correct behavior: Explain what a computer is.

User: "What is gravity?"
Correct behavior: Explain gravity.

User: "Why is the sky blue?"
Correct behavior: Explain Rayleigh scattering and the atmosphere.

NEVER answer a normal general question with:
"This question isn't related to Dipesh Kharel..."
"Please stay in context!"
"Happy to keep things focused..."

Those are refusal patterns and must NEVER be generated as the
answer to a legitimate general question.

The CURRENT USER MESSAGE HAS PRIORITY over previous conversation
messages. If an earlier assistant response was a refusal, fallback,
joke, or unrelated answer, do not copy it simply because it appears
in conversation history — re-evaluate the current question
independently.

Only refuse content that you would legitimately refuse for any
user, such as harmful, illegal, or explicit requests.

------------------------------------------------------------
IDENTITY
------------------------------------------------------------

- Dipesh Kharel is a Developer, Writer, and Visionary based in Kathmandu, Nepal.
- He is the Founder & CEO of NEXUS and Aivoke.Ai.
- He is currently a Grade 12 Science student at Cosmic International Academy.
- He works on AI platforms and web projects.
- He writes about technology, philosophy and society.

------------------------------------------------------------
TONE
------------------------------------------------------------

Witty, smooth, intelligent and natural.

For factual or technical questions: ANSWER THE QUESTION FIRST.
Personality is secondary to correctness.
Do not force Dipesh-related references into unrelated questions.

------------------------------------------------------------
REFERENCE MATERIAL
------------------------------------------------------------

The following material comes from the site's DKAI knowledge base.
Use it for questions actually related to Dipesh and his work.
It is NOT a restriction on your general knowledge.

${KB_REFERENCE || '(No DKAI reference material loaded.)'}

`.trim();


/* ============================================================
   GEMINI INITIALIZATION (@google/genai)
   ============================================================ */

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

console.log(
  ENABLE_SEARCH_GROUNDING
    ? '🔎 Google Search grounding: ON'
    : '🔎 Google Search grounding: OFF'
);

/*
 * Creates a fresh chat session for a single request.
 * `grounded` toggles the googleSearch tool for this session.
 */
function createChatSession(history, grounded) {
  return ai.chats.create({
    model: MODEL_NAME,
    history,
    config: {
      systemInstruction: DKAI_SYSTEM_PROMPT,
      maxOutputTokens: 64000,
      temperature: 0.75,
      tools: grounded ? [{ googleSearch: {} }] : undefined
    }
  });
}


/**
 * Builds Gemini `history` entries from the frontend's dkaiHistory array.
 * Only well-formed { role, text } entries survive.
 */
function toGeminiHistory(history) {
  if (!Array.isArray(history)) {
    return [];
  }

  return history
    .filter(
      (m) =>
        m &&
        typeof m.text === 'string' &&
        m.text.trim() &&
        (m.role === 'user' || m.role === 'bot' || m.role === 'model')
    )
    .slice(-20)
    .map((m) => ({
      role: m.role === 'user' ? 'user' : 'model',
      parts: [{ text: String(m.text).slice(0, 4000) }]
    }));
}


/* ============================================================
   EXPRESS
   ============================================================ */

const app = express();

app.set('trust proxy', true);

/*
 * CORS: open for initial testing. Tighten later by setting
 * ALLOWED_ORIGIN to your frontend's exact origin, e.g.
 * https://www.dipeshkharel14.com.np
 */
const allowedOrigin = process.env.ALLOWED_ORIGIN;

app.use(
  allowedOrigin
    ? cors({ origin: allowedOrigin })
    : cors({ origin: '*' })
);

app.use(express.json({ limit: '1mb' }));


/* ============================================================
   ROOT (simple sanity check)
   ============================================================ */

app.get('/', (req, res) => {
  res.send('DK AI Gemini backend is running.');
});


/* ============================================================
   RATE LIMITER
   ============================================================ */

const RATE_LIMIT_WINDOW_MS = 60 * 1000;
const RATE_LIMIT_MAX = 30;
const hits = new Map();

function rateLimited(ip) {
  const now = Date.now();

  const arr = (hits.get(ip) || []).filter(
    (t) => now - t < RATE_LIMIT_WINDOW_MS
  );

  arr.push(now);
  hits.set(ip, arr);

  return arr.length > RATE_LIMIT_MAX;
}


/* ============================================================
   HEALTH CHECKS
   ============================================================ */

app.get('/health', (req, res) => {
  res.send('DK AI Server is running!');
});

app.get('/api/health', (req, res) => {
  res.json({ ok: true, model: MODEL_NAME });
});


/* ============================================================
   DEBUG
   ============================================================ */

app.get('/api/debug', (req, res) => {
  res.json({
    ok: true,
    model: MODEL_NAME,
    port: PORT,
    hasGeminiKey: !!process.env.GEMINI_API_KEY,
    hasOpenAIKey: !!process.env.OPENAI_API_KEY,
    allowedOrigin: allowedOrigin || null,
    enableSearchGrounding: ENABLE_SEARCH_GROUNDING,
    ip: req.ip
  });
});


/* ============================================================
   GEMINI CONNECTIVITY TEST (debugging only)
   ============================================================ */

app.get('/api/test-gemini', async (req, res) => {
  try {
    const response = await ai.models.generateContent({
      model: MODEL_NAME,
      contents: 'Reply with exactly: Gemini connection successful.'
    });

    res.json({
      ok: true,
      answer: (response.text || '').trim()
    });

  } catch (err) {
    const safeMsg = redactSecrets(err?.message || String(err));

    console.error('Gemini test request failed:', safeMsg);

    res.status(500).json({
      ok: false,
      error: safeMsg
    });
  }
});


/* ============================================================
   CHAT API — STREAMING (NDJSON)
   ============================================================ */

app.post('/api/chat', async (req, res) => {

  try {

    const ip = req.ip || req.connection?.remoteAddress || 'unknown';

    if (rateLimited(ip)) {
      return res.status(429).json({
        error: 'Too many requests — slow down a bit.'
      });
    }

    const { message, history } = req.body || {};

    if (!message || typeof message !== 'string' || !message.trim()) {
      return res.status(400).json({
        error: 'message is required'
      });
    }

    const trimmedMessage = message.slice(0, 2000);
    const geminiHistory = toGeminiHistory(history);

    /*
     * Streaming NDJSON response — one JSON object per line.
     * Headers are only sent once the first byte is written, so a
     * failure before that point can still return a normal HTTP error
     * status (the frontend treats !res.ok as "fall back to offline
     * data" rather than trying to read a body).
     */
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('X-Accel-Buffering', 'no');

    async function streamFrom(grounded) {

      const chat = createChatSession(geminiHistory, grounded);

      const stream = await chat.sendMessageStream({
        message: trimmedMessage
      });

      let full = '';
      let lastChunk = null;

      for await (const chunk of stream) {

        const piece = chunk.text;

        if (piece) {
          full += piece;

          res.write(
            JSON.stringify({ type: 'chunk', text: piece }) + '\n'
          );
        }

        lastChunk = chunk;
      }

      return { full, lastChunk };
    }

    let streamed;
    let usedFallback = false;

    try {

      streamed = await streamFrom(ENABLE_SEARCH_GROUNDING);

    } catch (primaryErr) {

      /*
       * If grounding causes the request to fail and nothing has been
       * written to the client yet, retry once without grounding.
       */
      if (ENABLE_SEARCH_GROUNDING && !res.headersSent) {

        console.warn('⚠️ Grounded Gemini request failed.');
        console.warn(
          'Retrying without search grounding:',
          redactSecrets(primaryErr.message)
        );

        streamed = await streamFrom(false);
        usedFallback = true;

      } else {
        throw primaryErr;
      }
    }

    const grounding =
      usedFallback
        ? null
        : streamed.lastChunk?.candidates?.[0]?.groundingMetadata;

    const sources = (grounding?.groundingChunks || [])
      .map((c) =>
        c.web ? { title: c.web.title, uri: c.web.uri } : null
      )
      .filter(Boolean);

    res.write(
      JSON.stringify({
        type: 'done',
        sources,
        searched: sources.length > 0
      }) + '\n'
    );

    res.end();

  } catch (err) {

    const safeMsg = redactSecrets(err?.message || String(err));

    console.error('Gemini request failed:', safeMsg);

    if (res.headersSent) {
      res.write(
        JSON.stringify({ type: 'error', error: safeMsg }) + '\n'
      );
      res.end();
    } else {
      res.status(500).json({ error: safeMsg });
    }
  }
});


/* ============================================================
   START SERVER
   ============================================================ */

app.listen(PORT, () => {
  console.log(`✅ DK AI backend running on port ${PORT}`);
  console.log(`   Health check:  /api/health`);
  console.log(`   Debug:         /api/debug`);
  console.log(`   Gemini test:   /api/test-gemini`);
  console.log(`   Chat API:      /api/chat`);
});
