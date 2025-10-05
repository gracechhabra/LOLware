const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(express.static('public'));

app.post('/roast', (req, res) => {
  const code = req.body.code || "";
  let roasts = [];

 // Rule 1: Long function (>30 lines)
if (code.split('\n').length > 30) {
  roasts.push("This function is longer than a Bollywood movie. Grab popcorn 🍿");
}

// Rule 2: One-letter variables
if (/[ \t](x|y|z)[\s=;]/.test(code)) {
  roasts.push("Ah yes, x, y, and z — truly the holy trinity of confusion.");
}

// Rule 3: Too many console.log / print statements (>3)
if ((code.match(/console\.log|print/g) || []).length > 3) {
  roasts.push("This isn’t debugging, it’s a full-blown confessional.");
}

// Rule 4: No comments
if (!/\/\//.test(code) && !/\/\*/.test(code) && !/#/.test(code)) {
  roasts.push("Silent but deadly — no comments found. Future you is crying.");
}

// Rule 5: Too many TODOs
if ((code.match(/TODO/g) || []).length > 2) {
  roasts.push("Looks like you’re writing a wish list, not code.");
}

// Rule 6: Global variables (JS / Python)
if (/^\s*(var|let|const|global)/gm.test(code)) {
  roasts.push("Global variables? More like ticking time bombs 💣");
}

// Rule 7: Nested loops > 3
if ((code.match(/for\s*\(|while\s*\(/g) || []).length > 3) {
  roasts.push("This loop is so deep, you might find oil down there.");
}

// Rule 8: Bad indentation (mixed tabs and spaces)
if (/\t/.test(code) && / {2,}/.test(code)) {
  roasts.push("Mixed tabs and spaces… is this code or abstract art?");
}

// Rule 9: Empty functions
if (/function\s+\w*\s*\([^)]*\)\s*{\s*}/.test(code) || /def\s+\w*\([^)]*\):\s*pass/.test(code)) {
  roasts.push("Empty function? Was this a placeholder or procrastination?");
}

// Rule 10: Magic numbers (numbers not 0 or 1)
if (/\b[2-9][0-9]*\b/.test(code)) {
  roasts.push("Magic numbers everywhere! Are we coding or casting spells?");
}

// Rule 11: Single-line code (super short scripts)
if (code.split('\n').length <= 2) {
  roasts.push("Two lines of code… barely enough to wake the compiler.");
}

// Rule 12: Repeated variable names (x1, x2, x3…)
if (/(x\d+)/.test(code)) {
  roasts.push("Naming variables x1, x2… we get it, you like numbers.");
}

  // Fallback roast if nothing matched
  if (roasts.length === 0) {
    roasts.push("Surprisingly clean... suspiciously clean. Did ChatGPT write this?");
  }

  res.json({ roasts });
});

app.listen(PORT, () => {
  console.log(`Code Roaster running on http://localhost:${PORT}`);
});
