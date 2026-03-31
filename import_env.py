import os
from dotenv import load_dotenv

load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY")
quadrant_url = os.getenv("QUADRANT_URL")
quadrant_api_key = os.getenv("QUADRANT_API_KEY")

env_vars = {
    "GROQ_API_KEY": groq_api_key,
    "QUADRANT_URL": quadrant_url,
    "QUADRANT_API_KEY": quadrant_api_key,
}

for name, value in env_vars.items():
    if value:
        print(f"{name} loaded successfully.")
    else:
        print(f"Could not find {name} in the .env file.")

if not all(env_vars.values()):
    print("\nPlease create a .env file with:")
    print("GROQ_API_KEY=your_key_here")
    print("QUADRANT_URL=your_url_here")
    print("QUADRANT_API_KEY=your_key_here")