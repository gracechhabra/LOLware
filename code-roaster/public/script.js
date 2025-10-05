async function roastCode() {
  const code = document.getElementById('codeInput').value;

  const response = await fetch('/roast', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code })
  });

  const data = await response.json();
  const output = document.getElementById('output');
  output.innerHTML = data.roasts.map(r => `<p>👉 ${r}</p>`).join('');
}
