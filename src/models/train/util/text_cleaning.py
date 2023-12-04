import re
import numpy as np
import pandas as pd

def cleaning_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z가-힣0-9\s]', '', text)
    elif pd.isna(text):
        text = ''  # Handle NaN explicitly
    else:
        text = str(text)

    return text

