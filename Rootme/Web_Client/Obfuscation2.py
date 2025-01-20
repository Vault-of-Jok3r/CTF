import urllib.parse

# Chaîne encodée
encoded_string = "unescape%28%22String.fromCharCode%2528104%252C68%252C117%252C102%252C106%252C100%252C107%252C105%252C49%252C53%252C54%2529%22%29"

# Première étape : Décodage de l'encodage URL
decoded_once = urllib.parse.unquote(encoded_string)
decoded_twice = urllib.parse.unquote(decoded_once)

# Nettoyer et extraire les valeurs numériques de String.fromCharCode
import re

match = re.search(r"String\.fromCharCode\((.*?)\)", decoded_twice)
if match:
    char_codes = match.group(1)  # Extraire les codes entre parenthèses
    char_codes = [int(code.strip()) for code in char_codes.split(',')]
    result = ''.join(chr(code) for code in char_codes)
    print("Résultat final :", result)
else:
    print("Impossible de décoder complètement.")
