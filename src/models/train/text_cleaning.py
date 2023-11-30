import re

def cleaning_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z가-힣0-9\s]', '', text)
    return text