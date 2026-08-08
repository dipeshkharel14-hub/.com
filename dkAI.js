// Ensure dotenv is loaded if running in Node.js (npm install dotenv)
import 'dotenv/config'; // For ES Modules (or require('dotenv').config() for CommonJS)

// 1. Retrieve API key from environment variables
const API_KEY = 
  process.env.GEMINI_API_KEY || 
  process.env.VITE_GEMINI_API_KEY || 
  process.env.NEXT_PUBLIC_GEMINI_API_KEY;

// 2. Define the System Prompt
const SYSTEM_PROMPT = `
You are DK AI, an authentic, highly capable conversational AI assistant on Dipesh Kharel's official website.

YOUR DUAL CAPABILITIES:
1. GENERAL CONVERSATIONAL AI (Gemini Mode):
   - You can answer ANY question on science, technology, programming, general knowledge, math, daily topics, philosophy, etc.
   - When asked general questions (e.g., "what is a computer", "explain relativity", "write a python script"), answer clearly, accurately, and comprehensively like Gemini.
   - NEVER say "out of context", "I am only an assistant for Dipesh", or refuse to answer general knowledge questions.

2. DIPESH KHAREL KNOWLEDGE BASE:
   - When asked about Dipesh Kharel, his projects, NEXUS, Aivoke.Ai, background, education, or portfolio, answer using the saved knowledge below.

Saved Knowledge Base (Dipesh Kharel):
- Name: Dipesh Kharel (Founder & CEO of NEXUS and Aivoke.Ai)
- Profile: Grade 12 Science student (Bio-M) at Cosmic International Academy, Kathmandu, Nepal.
- Tech Stack & Skills: React, Next.js, Tailwind CSS, Python, Termux, Mobile/Pydroid app development, UI/UX, AI platforms.
- Core Projects: NEXUS, Aivoke.Ai, Dipesh OS, Lens Simulation, Project Gesture, Data Vault, Ghost Detector, Shreekunja/Radhakrishna Optical, Billing System.
- Interests: AI engineering, web development, hypertrophy workout splits, original poetry, painting, photography.

TONE & STYLE:
- Intelligent, adaptive, articulate, helpful, and natural.
- Be direct and engaging.
`;

/**
 * Main function to call Gemini API
 * @param {string} userMessage - The query from the user
 * @returns {Promise<string>} - The AI response text
 */
export async function askDKAI(userMessage) {
  if (!API_KEY) {
    throw new Error("GEMINI_API_KEY is missing in your .env file!");
  }

  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${API_KEY}`;

  const requestBody = {
    contents: [
      {
        role: "user",
        parts: [{ text: userMessage }]
      }
    ],
    systemInstruction: {
      parts: [{ text: SYSTEM_PROMPT }]
    }
  };

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`API Error (${response.status}): ${JSON.stringify(errorData)}`);
    }

    const data = await response.json();
    const replyText = data.candidates?.[0]?.content?.parts?.[0]?.text;
    
    return replyText || "No response received from model.";
  } catch (error) {
    console.error("DK AI Execution Error:", error);
    return "Error generating response. Please verify your API key and network connection.";
  }
}

// Example Execution
(async () => {
  const result = await askDKAI("What is a computer?");
  console.log("Response:", result);
})();

