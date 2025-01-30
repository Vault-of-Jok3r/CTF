import os
import crypt

PID = os.getpid() + 1
print(crypt.crypt(str(PID), "$1$awesome"))