import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io.wavfile as wf
import huffmancodec as h

def huffman(data, contador):
    data= np.asarray(data)
    media = 0
    variancia = 0
    data = data.flatten()
    soma = int(np.sum(contador))

    prob=[]
    np.asarray(prob)

    for j in contador: 
        x = j/soma
        if x!=0:
            prob.append(x)

    codec = h.HuffmanCodec.from_data(data)
    A,comp = codec.get_code_len() 

    if(len(prob) == len(comp)):
        for a in range (len(prob)):
            media += (prob[a] * comp[a])

        cmp=comp
        for e in range (len(comp)):
            cmp[e]=(comp[e] - media)**2

        variancia = np.average(cmp,  weights = prob)

    print("Media: " , media)
    print("Variancia: ", variancia)

    return media, variancia
        
# Constrói um histograma
def histograma(A,contador):       
    
    plt.bar(A,contador)
    plt.ylabel("Counter")
    plt.xlabel("Alphabet")
    plt.title("Ocorrencias")
    plt.show()
    
def contador(A,P):
   
    l = list(P)
    cont = []
    cont.asarray()
    
    for a in A:
        cont.append(l.count(a))
    
    np.asarray(cont)
    return cont        

# Calcula a entropia
def entropy(contador):
    
    entropia = 0
    prob= np.zeros(len(contador))
  
    prob= contador/ np.sum(contador)
    
    for i in prob:
        if(i>0):
            entropia += (i*np.log2(1/i))
            
    return entropia

def testImg(img_name):
    
    img = mpimg.imread(img_name)
    img2 = img.flatten()

    A, contador= np.unique(img2, return_counts = True)
    histograma(A, contador)
    print("Entropia: ", entropy(contador))
    huffman(img2, contador)
    agrupar(img2)

def testAudio(audio_name):
    inf = wf.read(audio_name)[1]
    if (inf.ndim == 2):
        data = inf[:,0]
        
    else:
        data = inf
    A, contador= np.unique(data, return_counts = True)
    
    print("Entropia:",entropy(contador))
    histograma(A,contador)
    huffman(data, contador)
    agrupar(data)
    
#função para leitura do texto
def lerTexto():
    fich = "english.txt"
    file = open(fich,'r')
    data = np.array(file.read().split())
    texto(data)
    file.close()
         
def texto(data):
    lista = []
    A = np.empty(52, dtype = np.str)
    contador= np.zeros(52, dtype = np.int32)
    
    for i in range(len(A)):
      if i<=25:
        A[i]= chr(i+65)
      else:
          A[i]=chr(i+71)

    for palavra in data:
        for letra in palavra:
            letter = ord(letra)
            #print(letter)
            if (letter > 64 and letter < 91):
                contador[letter-65] += 1
                lista.append(letter)
            elif(letter > 96 and letter < 123):
                contador[letter-71] += 1
                lista.append(letter)
    
    print("Entropia:",entropy(contador))
    huffman(lista,contador)
    agrupar(lista)
    histograma(A,contador)
     
def agrupar(data):
    
    alfa = np.arange(256)
    data = np.asarray(data)
    data = data.flatten()
    
    if ((data.size)%2 == 0):
        nlinhas = int((data.size)/2)
        agrupado=data.reshape(nlinhas,2)
    else:
        nlinhas = int((data.size-1)/2)
        agrupado=data[:-1].reshape(nlinhas,2)
        
    A = np.empty(nlinhas)
    for i in np.arange(nlinhas):
        A[i] = agrupado[i][0] *alfa.size + agrupado[i][1]
    
    contador = np.unique(A,return_counts=True)[1]
    
    nova_entropia = entropy(contador)
    print("Entropia agrupada:",nova_entropia/2, "\n")

def infoMutua(query, target, A):
    prob_m = probMutua(query,target,A)
    
    count_q = np.zeros(A.size)
    for i in np.nditer(query):
        count_q[i - A[0]] += 1    
    prob_q = count_q/query.size
    
    
    count_t = np.zeros(A.size)
    for j in np.nditer(target):
        count_t[j - A[0]] += 1
    prob_t = count_t/target.size
   
    mutual_info=0   
    
    for l in range(A.size):
        for k in range(A.size):
            if(prob_m[l,k] > 0 and prob_q[l] >0 and prob_t[k] > 0):
                frac = prob_m[l,k] / (prob_q[l] * prob_t[k])
                mutual_info += prob_m[l,k] * np.log2(frac)
                
    return mutual_info

def probMutua(query: np.ndarray, target:np.ndarray, A:np.ndarray):
    conta = np.zeros((256, 256))
    
    for i in range(query.size):
        conta[query[i] ,target[i]]+=1   
        
        
    return conta / (np.sum(conta))

def janela(query,target,A,passo):
       
    indices = np.arange(0, len(target)-len(query)+1, passo)
    count = np.zeros(indices.size)
      
    j=0
    for i in np.nditer(indices):
        section = target[i:i+len(query)]
        count[j]=infoMutua(query, section,A)
        j += 1
   
    return count

def infoMutua_Audio(query,target):
    info_query = wf.read(query)[1] 
    info_Target = wf.read(target)[1] 
        
    if (info_query.ndim == 2): 
        data = info_query[:,0]
        dataT = info_Target
    else:
        data = info_query
        dataT = info_Target
                
    minimo = min(np.min(data), np.min(dataT))
    maximo = max(np.max(data), np.max(dataT))
    
    step = int(data.size/4)   
    
    A = np.arange(minimo,maximo+1)  
    
    count = janela(data, dataT, A, step)
    
    print("Informação mútua de %s e de %s : %s\n" %(query, target,count))
    
    x_label = np.arange(1,count.size+1)
    plt.figure()
    plt.bar(x_label,count)
    plt.xlabel("Periodos")
    plt.ylabel("Informação mutua")
      
    return count

def maxi():
    maxim = np.zeros(7)
    for i in range(7):
        maxim[i] = max(infoMutua_Audio("./data/guitarSolo.wav","./data/Song0"+str(i+1)+".wav"))
    maximo= max(maxim)
    sort = -np.sort(-maxim)
    print("informação mutua de cada musica com o 'guitarSolo.wav' por ordem decrescente: %s \n" % sort)
    print("informação mutua máxima de todas as musicas comparando com o 'guitarSolo.wav': %s\n" % maximo)