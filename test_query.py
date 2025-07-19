import requests

API_URL = "http://localhost:8000/query"
query = "research gap of the paper?"

try:
    response = requests.post(API_URL, json={"query": query})
    response.raise_for_status()  # Raise error for bad HTTP status codes

    # Try to parse JSON
    data = response.json()
    print("✅ Response:", data)

except requests.exceptions.HTTPError as http_err:
    print(f"❌ HTTP error occurred: {http_err}")
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the server. Is FastAPI running at localhost:8000?")
except requests.exceptions.JSONDecodeError:
    print("❌ Response is not valid JSON. Raw response:")
    print(response.text)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
