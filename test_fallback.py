import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai import fallback_assistant
print("Loaded from:", fallback_assistant.__file__)

from ai.fallback_assistant import generate_fallback_answer
from dashboard.metrics import load_all_data

data = load_all_data()
result = generate_fallback_answer("security summary", data)

print("Return type:", type(result).__name__)
print("Repr:", repr(result)[:300])