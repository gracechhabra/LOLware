from pet_analyser import analyze_audio
from pet_translator import translate_pet_sound

result = analyze_audio("catmeow.wav")
print(result)
print("AI Translation:", translate_pet_sound(result))
