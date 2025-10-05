require('dotenv').config();
console.log("Loaded key:", !!process.env.GEMINI_API_KEY);

const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const PORT = 3000;

// ✅ Pass the key string, not an object
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ✅ Use valid model id + proper config key
const model = genAI.getGenerativeModel({
  model: 'gemini-2.5-flash',
  generationConfig: { temperature: 0.9, topP: 1, topK: 40, maxOutputTokens: 4096 }
});

app.use(express.json());
app.use(express.static('public'));

async function aiRoast(code) {
  const prompt = `Roast this code in a funny, witty way:\n${code}`;
  try {
    const result = await model.generateContent(prompt);
    return result.response.text(); // ✅ call the method
  } catch (error) {
    console.error('Gemini API error:', error);
    return 'AI roast failed… but your code is still questionable 😅 (Check your API key or model configuration!)';
  }
}

app.post('/roast', async (req, res) => {
  const code = req.body.code || '';
  let roasts = [];

  if (code.split('\n').length > 30)
    roasts.push('This function is longer than a Bollywood movie. Did you forget about scrolling?');
  if ((code.match(/console\.log|print/g) || []).length > 3)
    roasts.push('Too many console logs… this is a confessional, not code. Who are you debugging, exactly?');
  if (!/\/\//.test(code) && !/\/\*/.test(code) && !/#/.test(code))
    roasts.push('Silent but deadly — no comments found. Future you is going to hate present you.');
  if ((code.match(/TODO/g) || []).length > 2)
    roasts.push('Looks like you’re writing a wish list, not code. Maybe try getting to step 1 before writing step 10?');

  roasts.push(await aiRoast(code));
  res.json({ roasts });
});

app.listen(PORT, () => {
  console.log(`Code Roaster running on http://localhost:${PORT}`);
});