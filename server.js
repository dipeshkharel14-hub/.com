/**
 * DK AI — Gemini backend
 * Express server that proxies chat messages to Google's Gemini API.
 */

require('dotenv').config();

const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const PORT = process.env.PORT || 5000;

const MODEL_NAME =
  process.env.GEMINI_MODEL || 'gemini-2.5-flash';

const ENABLE_SEARCH_GROUNDING = false;


/* ============================================================
   API KEY
   ============================================================ */

if (!process.env.GEMINI_API_KEY) {
  console.error('\n❌ Missing GEMINI_API_KEY.');
  console.error(
    'Set GEMINI_API_KEY in your hosting provider environment variables.\n'
  );
  process.exit(1);
}

if (!/^AIzaSy/.test(process.env.GEMINI_API_KEY)) {
  console.warn(
    '⚠️ GEMINI_API_KEY does not start with "AIzaSy". Check your Gemini API key.'
  );
}


/* ============================================================
   LOAD DKAI_KB FROM INDEX.HTML
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

      const arrCode =
        script.slice(startIdx, endIdx + 2);

      const sandbox = {};

      // Trusted self-authored index.html content.
      new Function(
        'sandbox',
        arrCode +
          '\nsandbox.DKAI_KB = DKAI_KB;'
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
   MASTER GEMINI SYSTEM PROMPT
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
- current-information questions when web grounding is available

IMPORTANT:

A question DOES NOT need to be related to Dipesh Kharel,
NEXUS, or Aivoke.Ai.

If the user asks a normal general question, ANSWER IT DIRECTLY.

For example:

User:
"What is a computer?"

Correct behavior:
Explain what a computer is.

User:
"What is gravity?"

Correct behavior:
Explain gravity.

User:
"Why is the sky blue?"

Correct behavior:
Explain Rayleigh scattering and the atmosphere.

NEVER answer a normal general question with:

"This question isn't related to Dipesh Kharel..."

"Please stay in context!"

"Happy to keep things focused..."

Those are client-side fallback messages and must NEVER be generated
as the answer to a legitimate general question.

The CURRENT USER MESSAGE HAS PRIORITY over previous conversation
messages.

If an earlier assistant response was a refusal, fallback, joke,
or unrelated answer, DO NOT copy it simply because it appears
in conversation history.

Re-evaluate the current question independently.

Only refuse content that you would legitimately refuse for any user,
such as harmful, illegal, or explicit requests.

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

For factual or technical questions:
ANSWER THE QUESTION FIRST.

Personality is secondary to correctness.

Do not force Dipesh-related references into unrelated questions.

------------------------------------------------------------
RELATIONSHIP STATUS
------------------------------------------------------------

- Dipesh is 100% single.
- He is a one-woman type and an old-school gentleman.
- Do not invent contradictory relationship facts.

------------------------------------------------------------
SPECIAL PROJECT
------------------------------------------------------------

If asked about the "special project", explain that Dipesh built
a dedicated architecture with a specific person in mind, but the
slot was never selected or claimed.

------------------------------------------------------------
STRANGER POLICY
------------------------------------------------------------

Dipesh ignores cold DMs and random online strangers.

Do not encourage strangers to believe they have a real chance.

------------------------------------------------------------
NEGATIVE FEEDBACK
------------------------------------------------------------

If the user says "bad", "wrong", "boring", "sucks", etc.,
respond naturally and confidently.

------------------------------------------------------------
HARD BOUNDARIES
------------------------------------------------------------

Never invent facts about Dipesh that contradict the identity section.

Never claim to have real private system logs or surveillance data.

Never produce explicit sexual content, harassment,
or degrading content involving real people.

For questions about Dipesh/NEXUS/Aivoke.Ai:
keep answers concise unless detail is requested.

For real questions:
answer with whatever length is appropriate.

------------------------------------------------------------
REFERENCE MATERIAL
------------------------------------------------------------

The following material comes from the site's DKAI knowledge base.

Use it for questions actually related to Dipesh and his work.

It is NOT a restriction on your general knowledge.

${KB_REFERENCE || '(No DKAI reference material loaded.)'}

`.trim();


/* ============================================================
   GEMINI INITIALIZATION
   ============================================================ */

const genAI =
  new GoogleGenerativeAI(
    process.env.GEMINI_API_KEY
  );

const model =
  genAI.getGenerativeModel({
    model: MODEL_NAME,
    systemInstruction: DKAI_SYSTEM_PROMPT,

    tools: ENABLE_SEARCH_GROUNDING
      ? [{ googleSearch: {} }]
      : undefined
  });

const modelPlain =
  ENABLE_SEARCH_GROUNDING
    ? genAI.getGenerativeModel({
        model: MODEL_NAME,
        systemInstruction: DKAI_SYSTEM_PROMPT
      })
    : model;

console.log(
  ENABLE_SEARCH_GROUNDING
    ? '🔎 Google Search grounding: ON'
    : '🔎 Google Search grounding: OFF'
);


/* ============================================================
   EXPRESS
   ============================================================ */

const app = express();

app.set('trust proxy', true);

const allowedOrigin =
  process.env.ALLOWED_ORIGIN;

app.use(
  allowedOrigin
    ? cors({ origin: allowedOrigin })
    : cors()
);

app.use(
  express.json({
    limit: '1mb'
  })
);


/* ============================================================
   ROOT
   ============================================================ */

app.get('/', (req, res) => {
  res.sendFile(
    path.join(__dirname, 'index.html')
  );
});


/* ============================================================
   RATE LIMITER
   ============================================================ */

const RATE_LIMIT_WINDOW_MS =
  60 * 1000;

const RATE_LIMIT_MAX = 30;

const hits = new Map();

function rateLimited(ip) {
  const now = Date.now();

  const arr =
    (hits.get(ip) || [])
      .filter(
        (t) =>
          now - t <
          RATE_LIMIT_WINDOW_MS
      );

  arr.push(now);

  hits.set(ip, arr);

  return arr.length >
    RATE_LIMIT_MAX;
}


/* ============================================================
   GEMINI HISTORY
   ============================================================ */

function toGeminiHistory(history) {

  if (!Array.isArray(history)) {
    return [];
  }

  /*
   * These are artificial frontend fallback messages.
   * They should NEVER become conversational context for Gemini.
   */

  const blockedAssistantReplies = [
    "This question isn't related to Dipesh Kharel, NEXUS, Aivoke.Ai, or his work. Please stay in context!",

    "Happy to keep things focused — ask me anything about Dipesh's background, NEXUS, Aivoke.Ai, his projects, or his writing!",

    "I couldn't reach Gemini right now. Please try that question again in a moment."
  ];

  return history

    .filter(
      (m) =>
        m &&
        typeof m.text === 'string' &&
        (
          m.role === 'user' ||
          m.role === 'bot'
        )
    )

    /*
     * Always preserve user messages.
     * Remove artificial fallback bot messages.
     */

    .filter(
      (m) =>
        m.role === 'user' ||
        !blockedAssistantReplies.some(
          (bad) =>
            String(m.text).trim() === bad
        )
    )

    .slice(-20)

    .map((m) => ({
      role:
        m.role === 'user'
          ? 'user'
          : 'model',

      parts: [
        {
          text:
            String(m.text)
              .slice(0, 4000)
        }
      ]
    }));
}


/* ============================================================
   CHAT API
   ============================================================ */

app.post(
  '/api/chat',
  async (req, res) => {

    try {

      const ip =
        req.ip ||
        req.connection?.remoteAddress ||
        'unknown';

      if (rateLimited(ip)) {

        return res
          .status(429)
          .json({
            error:
              'Too many requests — slow down a bit.'
          });
      }


      const {
        message,
        history
      } = req.body || {};


      if (
        !message ||
        typeof message !== 'string' ||
        !message.trim()
      ) {

        return res
          .status(400)
          .json({
            error:
              'message is required'
          });
      }


      const geminiHistory =
        toGeminiHistory(history);


      /*
       * Limit user message size.
       */

      const trimmedMessage =
        message.slice(0, 2000);


      /*
       * Streaming NDJSON response.
       */

      res.setHeader(
        'Content-Type',
        'text/plain; charset=utf-8'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache, no-transform'
      );

      res.setHeader(
        'X-Accel-Buffering',
        'no'
      );


      /* ========================================================
         GEMINI STREAM
         ======================================================== */

      async function streamFrom(
        activeModel
      ) {

        const chat =
          activeModel.startChat({
            history:
              geminiHistory,

            generationConfig: {
              maxOutputTokens: 900,
              temperature: 0.75
            }
          });


        const streamResult =
          await chat.sendMessageStream(
            trimmedMessage
          );


        let full = '';


        for await (
          const chunk
          of streamResult.stream
        ) {

          const piece =
            chunk.text();

          if (piece) {

            full += piece;

            res.write(
              JSON.stringify({
                type: 'chunk',
                text: piece
              }) + '\n'
            );
          }
        }


        const finalResponse =
          await streamResult.response;


        return {
          full,
          finalResponse
        };
      }


      /* ========================================================
         PRIMARY GEMINI REQUEST
         ======================================================== */

      let streamed;

      let usedFallback = false;


      try {

        streamed =
          await streamFrom(model);

      } catch (primaryErr) {

        /*
         * If Google Search grounding causes the request to fail,
         * retry once without grounding.
         */

        if (
          model !== modelPlain &&
          !res.headersSent
        ) {

          console.warn(
            '⚠️ Grounded Gemini request failed.'
          );

          console.warn(
            'Retrying without search grounding:',
            primaryErr.message
          );


          streamed =
            await streamFrom(
              modelPlain
            );

          usedFallback = true;

        } else {

          throw primaryErr;
        }
      }


      /* ========================================================
         SEARCH SOURCES
         ======================================================== */

      const grounding =
        usedFallback
          ? null
          : streamed
              .finalResponse
              .candidates?.[0]
              ?.groundingMetadata;


      const sources =
        (
          grounding?.groundingChunks ||
          []
        )

          .map(
            (c) =>
              c.web
                ? {
                    title:
                      c.web.title,

                    uri:
                      c.web.uri
                  }
                : null
          )

          .filter(Boolean);


      res.write(
        JSON.stringify({
          type: 'done',

          sources,

          searched:
            sources.length > 0
        }) + '\n'
      );


      res.end();


    } catch (err) {
  console.error("GEMINI ERROR:", err);

  if (res.headersSent) {
    res.write(JSON.stringify({
      type: "error",
      error: err?.message || String(err)
    }) + "\n");
    res.end();
  } else {
    res.status(500).json({
      error: err?.message || String(err)
    });
  }
    }

      console.error(
        err?.stack || ''
      );


      const clientMsg = err?.message || String(err);
      /*
       * If streaming has already started,
       * send an error line instead of attempting
       * to change the HTTP status.
       */

      if (res.headersSent) {

        res.write(
          JSON.stringify({
            type: 'error',
            error: clientMsg
          }) + '\n'
        );

        res.end();

      } else {

        res
          .status(500)
          .json({
            error: clientMsg
          });
      }
    }
  }
);


/* ============================================================
   HEALTH CHECK
   ============================================================ */

app.get(
  '/api/health',
  (req, res) => {

    res.json({
      ok: true,
      model: MODEL_NAME
    });
  }
);


/* ============================================================
   SIMPLE HEALTH
   ============================================================ */

app.get(
  '/health',
  (req, res) => {

    res.send(
      'DK AI Server is running!'
    );
  }
);


/* ============================================================
   DEBUG
   ============================================================ */

app.get(
  '/api/debug',
  (req, res) => {

    res.json({

      ok: true,

      model:
        MODEL_NAME,

      port:
        PORT,

      hasGeminiKey:
        !!process.env.GEMINI_API_KEY,

      allowedOrigin:
        process.env.ALLOWED_ORIGIN ||
        null,

      enableSearchGrounding:
        ENABLE_SEARCH_GROUNDING,

      ip:
        req.ip
    });
  }
);


/* ============================================================
   START SERVER
   ============================================================ */

app.listen(
  PORT,
  () => {

    console.log(
      `✅ DK AI backend running on port ${PORT}`
    );

    console.log(
      `   Health check: /api/health`
    );

    console.log(
      `   Chat API: /api/chat`
    );
  }
);
