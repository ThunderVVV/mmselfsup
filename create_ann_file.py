import os

file=open('data/synth.txt',mode='w')
imgs = sorted(os.listdir("data/synth"))
for img in imgs:
    print(img)
    file.write(img + "\n")
file.close()
