import re
import sympy

def clean_math_text(text):
    # Standardize LaTeX delimiters
    text = re.sub(r'\$(.*?)\$', r'<|equation|>\1<|endequation|>', text)
    
    # Simplify notations
    text = text.replace("\\times", "×").replace("\\div", "÷")
    
    # Validate equations using SymPy
    try:
        equations = re.findall(r'<\|equation\|>(.*?)<\|endequation\|>', text)
        for eq in equations:
            sympy.sympify(eq)
    except:
        return None  # Filter invalid equations
    
    return text