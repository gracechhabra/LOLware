import google.generativeai as genai

# 🔑 configure Gemini
genai.configure(api_key="AIzaSyAYVhwV4fMrI5MWGKwb7o84GX8xYB2we6Y")

# ✅ test a simple text generation
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("Say hello! How are you today?")
print("Gemini says:", response.text)

