import xlwt
from xlwt import Workbook

wb = Workbook()

sheet = wb.add_sheet('Valsheet')
bold = xlwt.easyxf('font: bold 1')
bluebold = xlwt.easyxf('font: bold 1, color blue;')
f = open("result.noclass",'r')
dic = {}
for ln in f:
	line = ln.strip()
	if line == "":
		continue
	if line[0] == 'E':
		continue
	tups = [x for x in line.split() if len(x)>=2 and (x[-2]=='h' or x[-2]=='x')]
	if len(tups)==0:
		#find the class model type and iteration num
		# classmodel,iterationnum,typegen = [('.'.join(x.split('.')[4:-2]),int(x.split('.')[-2]),x.split('.')[-1]) for x in line.split() if x[0]=='g'][0]
		classmodel,iterationnum,typegen = [('.'.join(x.split('.')[4:-3]),int(x.split('.')[-3]),x.split('.')[-2]) for x in line.split() if x[0]=='g'][0]
		tup = (classmodel,iterationnum,typegen)
	else:
		tup=(tups[0],0,'baseline')
	# print(classmodel,iterationnum)
	if dic.get(tup) is None:
		dic[(tup)]={}
	if line[0]=='e':
		#edit distance
		dic[tup]['edit'] = float(line.split()[-1])
	if line[0]=='(':
		#fk,ts,ds
		dic[tup]['fk'] = float(line.split()[-3])
		dic[tup]['ts'] = float(line.split()[-2])
		dic[tup]['ds'] = float(line.split()[-1])
	if line[0]=='S':
		dic[tup]['SARI'] = float(line.split()[3])
	if line[0]=='B':
		dic[tup]['BLEU'] = float(line.split()[3])
	if line[0]=='i':
		dic[tup]['iBLEU'] = float(line.split()[3])
	if line[0]=='f':
		dic[tup]['fkBLEU'] = float(line.split()[3])
	if line[0]=='w':
		dic[tup]['worddiff'] = float(line.split()[3])
print(dic)

dic = dic.items()
for idx,(key,val) in enumerate(list(dic)):
	if idx==0:
		columntitles = ['modelname','modelnum','gen-type']+[k for k in val]
		for i,colname in enumerate(columntitles):
			sheet.write(0,i,colname)
	columnvalues = [key[0],key[1],key[2]]+[vl for ky,vl in val.items()]
	for i,colval in enumerate(columnvalues):
		sheet.write(idx+1,i,colval)


wb.save("sample2.xls")