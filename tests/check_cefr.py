
try:
    from cefrpy import CEFRAnalyzer
    analyzer = CEFRAnalyzer()
    print(f"Attributes: {dir(analyzer)}")
except Exception as e:
    print(f"Error: {e}")
