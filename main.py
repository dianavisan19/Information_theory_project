# -*- coding: utf-8 -*-
import functions as f

print("| Testes Imagens------------------------------|\n")
print("| Imagem: kid.bmp-----------------------------|\n")

f.testImg("./data/kid.bmp")

print("| Imagem: homer.bmp---------------------------|\n")

f.testImg("./data/homer.bmp")

print("| Imagem: homerBin.bmp------------------------|\n")

f.testImg("./data/homerBin.bmp")


print("| Testes Texto--------------------------------|\n")
print("| Ficheiro Texto: english.txt-----------------|\n")

f.lerTexto() 


print("| Testes Audio--------------------------------|\n")
print("| Ficheiro Audio: guitarSolo.wav--------------|\n")

f.testAudio("./data/guitarSolo.wav")
f.infoMutua_Audio("./data/guitarSolo.wav","./data/target01 - repeat.wav")
f.infoMutua_Audio("./data/guitarSolo.wav","./data/target02 - repeatNoise.wav")
f.maxi()

print("| Fim dos testes------------------------------|\n")