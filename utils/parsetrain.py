import matplotlib
import matplotlib.pyplot as plt

f = open("logs/TSdecembedd/out.wgan.allclass.denoi.back1.singleclassf.rho1.0.10k",'r')

celoss = []
discloss = []
classloss = []
genloss = []
for ln in f:
	if ln == "\n":
		continue
	line =ln.strip().split()
	if line[0]=='BATCH':
		if 'celoss' in line[1] :
			val = line[-1]
			celoss.append(tuple(float(x) for x in val.split(','))) #(celoss,pplperword)
		elif 'disc' in line[3]:
			val = line[-4:]
			discloss.append(tuple(float(x) for x in val)) #((recsim,norsim,norcom,reccom))
		elif 'class' in line[3]:
			val = line[-2:]
			classloss.append(tuple(float(x) for x in val)) #(recsim,reccom)
		elif 'gener' in line[3]:
			val = line[-4:]
			genloss.append(tuple(float(x) for x in val)) #(discnorsim,discnorcom,classnorsim,classnorcom)

# print(celoss)
# print()
plt.plot(list(range(len(celoss))),list(list(zip(*celoss))), linewidth=1.0)

plt.show()
