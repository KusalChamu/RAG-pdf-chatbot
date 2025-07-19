import google.generativeai as genai

genai.configure(api_key="AIzaSyA7oSdx0lQEQQqGjoCuJk0U4Ame8cJH4L4")

for m in genai.list_models():
    print(m.name, " | ", m.supported_generation_methods)
