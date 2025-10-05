Code Roaster
Test 1: Long Function + Console Logs + One-letter variables
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


Expected roasts triggered:

Long function → “This function is longer than a Bollywood movie.”

One-letter variables → “Ah yes, x, y, and z — truly the holy trinity of confusion.”

Too many console.log → “This isn’t debugging, it’s a full-blown confessional.”

🔹 Test 2: No comments + Magic Numbers
def calculate():
    result = 42 * 7 + 99
    return result


Expected roasts triggered:

No comments → “Silent but deadly — no comments found. Future you is crying.”

Magic numbers → “Magic numbers everywhere! Are we coding or casting spells?”

🔹 Test 3: Empty Function + Too many TODOs
// TODO: implement login
// TODO: implement logout
// TODO: implement signup
function placeholder() {}


Expected roasts triggered:

Empty function → “Empty function? Was this a placeholder or procrastination?”

Too many TODOs → “Looks like you’re writing a wish list, not code.”

🔹 Test 4: Bad Indentation + Nested Loops
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


Expected roasts triggered:

Nested loops > 3 → “This loop is so deep, you might find oil down there.”

Bad indentation → “Mixed tabs and spaces… is this code or abstract art?”

🔹 Test 5: Single-line code
console.log("Hello World!");


Expected roast triggered:

Single-line code → “Two lines of code… barely enough to wake the compiler.”

🔹 Test 6: Repeated variable names
let x1 = 5;
let x2 = 10;
let x3 = 15;


Expected roast triggered:

Repeated variable names → “Naming variables x1, x2… we get it, you like numbers.”