import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
import pickle
    
class AESCipher(object):
    def __init__(self, key): 
        self.bs = 32
        self.key = hashlib.sha256(key.encode()).digest()
        
    def encrypt(self, parameters):
        flat_parameters = [y for x in parameters for y in x.flatten()]
        bytes = self._pad(pickle.dumps(flat_parameters))
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(bytes))
        
    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        bytes = self._unpad(cipher.decrypt(enc[AES.block_size:]))
        return pickle.loads(bytes)
        
    def _pad(self, s):
        ss = (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)
        return s + ss.encode("utf8")
        
    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]
        
        
