import numpy as np
import tenseal as ts
import time

#need to install:
# - cmake
# clone
# python -m pip install .


def set_contexts(N,poly_mod_degree,coeff_mod_bit_sizes, scale):
    contexts=[]
    skeys=[]
    pkeys=[]
    ctx = ts.context(ts.SCHEME_TYPE.MK_CKKS,poly_mod_degree,-1,coeff_mod_bit_sizes)
    sk = ctx.secret_key()
    ctx.generate_galois_keys(sk)
    ctx.make_context_public()
    ctx.global_scale = scale
    contexts.append(ctx)
    skeys.append(sk)
    pk = ctx.public_key()
    pkeys.append(pk)
    for _ in range(N-1):
        ctx_eval = ts.context(ts.SCHEME_TYPE.MK_CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes,public_key=pk)
        sk = ctx_eval.secret_key()
        ctx_eval.generate_galois_keys(sk)
        ctx_eval.make_context_public()
        ctx_eval.global_scale = scale
        contexts.append(ctx_eval)
        skeys.append(sk)
        pk = ctx_eval.public_key()
        pkeys.append(pk)
    return contexts,skeys,pkeys

def generate_plaintexts(N,l,s):
    plain_vectors=[]
    for i in range(N):
        v=[]
        for j in range(l):
            v.append(np.random.randint(int(200_000_001/20)))
            #v.append(np.random.normal(0,s))
        plain_vectors.append(v)
    return plain_vectors

def aggregated_public_key(contexts):
    pk_sum = ts.ckks_vector(contexts[0],contexts[0].public_key())
    #print(type(contexts[0]),type(contexts[0].public_key()), type(pk_sum))
    #print("pk_sum ", contexts[0].public_key().data.data().data(0), contexts[0].public_key().data.data().data(1))
    #ctx_proto = contexts[0].serialize()
    #pk_proto = pk_sum.serialize()
    #ctx2 = ts.context_from(ctx_proto)
    #print("ctx2 pk ", ctx2.public_key().data.data().data(0), ctx2.public_key().data.data().data(1))
    #pk2 = ts.ckks_vector(ctx2,ctx2.public_key())
    #print("pk2 ", pk2.data.ciphertext()[0].data(0), pk2.data.ciphertext()[0].data(1))
    #print("pk2 ", pk2.data.data().data(0), pk2.data.data().data(1))
    for ctx in contexts[1:]:
        pk_sum=pk_sum.add_pk(ts.ckks_vector(ctx,ctx.public_key()))
    return ts._ts_cpp.PublicKey(pk_sum.data.ciphertext()[0])

def set_aggregated_public_key(contexts,pk):
    for ctx in contexts:
        ctx.data.set_publickey(pk)

def encrypt(contexts,plain_vectors):
    ctx = contexts[0]
    #v = ts.ckks_vector(ctx, plain_vectors[0])
    #print("v ", v.ciphertext()[0].data(0), v.ciphertext()[0].data(1))
    #ctx_proto= ctx.serialize()
    #v_proto = v.serialize()
    #ctx2=ts.context_from(ctx_proto)
    #v2=ts.ckks_vector_from(ctx2,v_proto)
    #print("v2 ", v2.ciphertext()[0].data(0), v2.ciphertext()[0].data(1))
    encrypted_vectors=[]
    for ctx,plain_vector in zip(contexts,plain_vectors):
        encrypted_vectors.append(ts.ckks_vector(ctx,plain_vector))
    return encrypted_vectors

def sum_vectors(vectors):
    sum=vectors[0]
    for vector in vectors[1:]:
        sum=sum+vector
    return sum

def get_shares(encrypted_sum,contexts,secret_keys):
    shares=[]
    for context,key in zip(contexts,secret_keys):
        shares.append(encrypted_sum.decryption_share(context,key))
    #print(type(shares[0]),shares)
    return shares

def decrypt_decode(c_sum, ds):
    c=c_sum
    for share in ds:
        c= c.add_share(share)
    return c.mk_decode()

def decrypt(c_sum, ds):
    c=c_sum
    return c.mk_decrypt(ds)

def encrypted_print(encrypted_vector):
    print("encrypted vector: ", end="")
    for c in encrypted_vector:#.data.ciphertext():
        for j in range(c.size()):
            print(c.data(j), end=" ")
    print("\n")

def close(vec1,vec2,diff):
    ko = f"The vectors are not close at positions: "
    avg=0
    max=0
    min=float('inf')
    close = True
    if len(vec1)!=len(vec2):
        return "The vectors have different lengths"
    for i in range(len(vec1)):
        d= abs(vec1[i]-vec2[i])
        avg += d
        if d> diff:
            close=False
            ko+= f"{i} "
        if d>max:
            max=d
        if d< min:
            min=d 
    avg/=len(vec1)       
    if close:
        return f"The vectors are close with difference {diff}, avg diff {avg}, min diff {min}, max diff {max}"
    ko+= f" , avg diff {avg}, min diff {min}, max diff {max}"
    return ko



#PARAMETERS#
poly_mod_degree =8192 #4096
coeff_mod_bit_sizes =[60,40,40,60]#[40,20,40]
scale=2**40 #2**20
N=3 #number of clients 10
l=1 #vectors length 30
s=10000 #std deviation
print(f"N: {N}, l: {l}, s: {s}")

#plaintexts
plain_vectors=generate_plaintexts(N,l,s)

x=1
times=np.zeros(x)
for i in range(x):
    t1=time.time()
    #CONTEXT#
    contexts,secret_keys,public_keys= set_contexts(N,poly_mod_degree,coeff_mod_bit_sizes,scale)

    #ENCRYPTION#
    #pk aggregation
    aggregated_pk = aggregated_public_key(contexts)
    set_aggregated_public_key(contexts,aggregated_pk)
    #encryption
    encrypted_vectors=encrypt(contexts,plain_vectors)

    #SUM
    encrypted_sum= sum_vectors(encrypted_vectors)

    #DECRYPTION#
    #decryption shares
    shares=get_shares(encrypted_sum,contexts,secret_keys)
    #decryption
    decrypted =decrypt(encrypted_sum,shares)
    t2=time.time()
    times[i]=t2-t1
    expected =np.sum(np.array(plain_vectors),axis=0)
    print("decryption ", decrypted)
    print("expected ", expected)
    print(close(decrypted,expected, 0.05))
    #incremental decryption
    decoded =decrypt_decode(encrypted_sum,shares)
    print("decryption ", decoded)
    print("expected ", expected)
    print(close(decoded,expected, 0.05))
print(times)
print(np.mean(times))
