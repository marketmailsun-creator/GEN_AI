import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

prompt = "Write a 4-line poem about Hyderabad in the monsoon."
response = model.generate_content(prompt)

print(response.text)
import langchainhub
print(dir(langchainhub))



