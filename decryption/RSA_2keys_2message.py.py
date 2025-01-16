from Crypto.PublicKey import RSA
from Crypto.Util.number import long_to_bytes, bytes_to_long
import gmpy2

# Extraire les modules et exposants des clés publiques
with open("public_key_1.pem", "r") as f:
    key1 = RSA.import_key(f.read())

with open("public_key_2.pem", "r") as f:
    key2 = RSA.import_key(f.read())

n = key1.n
e1 = key1.e
e2 = key2.e

# Lire les messages chiffrés
with open("message1", "r") as f:
    c1 = int(f.read().strip())

with open("message2", "r") as f:
    c2 = int(f.read().strip())

# Trouver les coefficients de Bézout pour e1 et e2
gcd, a, b = gmpy2.gcdext(e1, e2)

# Déchiffrer le message
m = (pow(c1, a, n) * pow(c2, b, n)) % n

# Convertir le message en bytes
message = long_to_bytes(m)
print(message.decode())