# Code Roaster 🔥

**Code Roaster** is a fun website that roasts your code every time you write something. Perfect for developers who need a laugh (or a reality check) while coding using GEMINI API. 

![Code Roast](https://img.shields.io/badge/Status-Active-brightgreen) ![Language](https://img.shields.io/badge/Language-JavaScript%20%7C%20Python-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Features](#features) 
- [Tech Stack](#techstack) 
- [Examples](#examples)  
- [Usage](#usage)  
- [License](#license)  

---

## Features 🎯

- ⚡ Detects **long functions**  
- 🤯 Calls out **one-letter variable names**  
- 📝 Spots **too many `console.log` statements**  
- 🪄 Warns about **magic numbers**  
- 🕳️ Flags **empty functions** and **too many TODOs**  
- 🎨 Notices **bad indentation** and **nested loops**  
- 💤 Roasts **single-line code** and **repeated variable names**  

---
## Tech Stack

- **AI/LLM:** Google Gemini (Gemini API)
- **Language:** JavaScript (ES6+)
- **Frontend:** HTML5, CSS3 (custom styling)
- **Optional tooling:** Node.js (for local dev/build), Vite/Webpack or plain static hosting

---
## Examples 💻

### Test 1: Long Function + Console Logs + One-letter variables
```javascript
function crazyFunction() {
  let x = 5;
  let y = 10;
  let z = x + y;
  console.log(x);
  console.log(y);
  console.log(z);
  console.log(x + y + z);
  console.log("Debugging");
  console.log("Another log");
  console.log("Yet another log");
  console.log("Keep logging");
  console.log("Last log");
  // Imagine this continues to make function >30 lines
}
```

### Test 2: No Comments + Magic Numbers
```python
def calculate():
    result = 42 * 7 + 99
    return result
```

### Test 3: Empty Function + Too many TODOs
```javascript
function placeholder() {}
```

### Test 4: Bad Indentation + Nested Loops
```javascript
function nested() {
    for (let i = 0; i < 5; i++) {
    for (let j = 0; j < 5; j++) {
        for (let k = 0; k < 5; k++) {
        for (let l = 0; l < 5; l++) {
            console.log(i,j,k,l);
        }
        }
    }
    }
}
```