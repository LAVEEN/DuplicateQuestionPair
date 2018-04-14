import sys

fin = sys.argv[1]

#fouttr = sys.argv[3]
#fouttt = sys.argv[4]

fwtr = open("train.csv", "w")
fwtt = open("test.csv", "w")


frtr = open(fin, "r")



lines = frtr.readlines()

tot_lines = 0



for j, line in enumerate(lines):
	if len(line) > 1:
		tot_lines += 1


tr_l = int(0.7 * tot_lines)
tt_l = tot_lines - tr_l


for j, line in enumerate(lines):
	if len(line)>1:
		line  = line.strip()
		if j==0:
			fwtr.write(line + '\n')
			fwtt.write(line + '\n')
		elif j <= tr_l:
			fwtr.write(line + '\n')
		else:
			fwtt.write(line + '\n')


