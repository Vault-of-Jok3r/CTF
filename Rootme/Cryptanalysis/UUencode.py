import binascii

try:
    with open("ch1.txt", "r") as file:
        uuencoded_data = file.read()
except FileNotFoundError:
    print("Fichier introuvable.")
    exit()

try:
    start = uuencoded_data.find("begin")
    end = uuencoded_data.find("end")

    if start == -1 or end == -1:
        print("Le fichier uuencodé n'est pas valide : 'begin' ou 'end' manquant.")
        exit()

    uuencoded_body = uuencoded_data[start:end].strip()

    uuencoded_lines = uuencoded_body.split('\n')
    uuencoded_body = '\n'.join(uuencoded_lines[1:])

except Exception as e:
    print(f"Erreur lors de l'extraction des données uuencodées : {e}")
    exit()

try:
    decoded_output = binascii.a2b_uu(uuencoded_body)
    print("Décodage réussi.")
    print(decoded_output.decode())
except binascii.Error as e:
    print("Erreur lors du décodage :", e)
except UnicodeDecodeError as e:
    print("Erreur lors de la conversion en texte :", e)