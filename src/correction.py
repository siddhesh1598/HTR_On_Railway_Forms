def numCheck(val):
	zero = ['D', 'O', 'o']
	one = [',', 'l', 'I', '/', 't']
	four = ['A']
	five = ['s']
	seven = ['T']
	eight = ['S']

	l = []
	for i in val:
		l.append(i)

	for i in range(len(l)):
		if l[i] in zero:
			l[i] = '0'
		if l[i] in one:
			l[i] = '1'
		if l[i] in four:
			l[i] = '4'
		if l[i] in five:
			l[i] = '5'
		if l[i] in seven:
			l[i] = '7'
		if l[i] in eight:
			l[i] = '8' 

	val = ("").join(l)

	return val


def correction(recognizedOutput):
	dates = ['date_dd', 'date_mm', 'date_yyyy', 
			'dob_dd', 'dob_mm', 'dob_yyyy',
			'dot_dd', 'dot_mm', 'dot_yyyy']

	for i in dates:
		x = recognizedOutput[i]
		x = x.replace(" ", "")

		if i == 'date_yyyy' or i == 'dot_yyyy':
			if len(x) == 2:
				x = "20" + x
			recognizedOutput[i] = x

		else:
			if len(x) == 1:
				x = "0" + x
			recognizedOutput[i] = x

	aadhar = recognizedOutput['aadhar']
	aadhar = aadhar.replace(" ", "")
	recognizedOutput['aadhar'] = aadhar

	num = ['date_dd', 'date_mm', 'date_yyyy', 
			'dob_dd', 'dob_mm', 'dob_yyyy',
			'mobile', 'aadhar', 'train', 
			'dot_dd', 'dot_mm', 'dot_yyyy']

	for n in num:
		recognizedOutput[n] = numCheck(recognizedOutput[n])

	newOutput = {}

	newOutput['Date'] = recognizedOutput["date_dd"] + '-' + recognizedOutput['date_mm'] + '-' + recognizedOutput['date_yyyy']
	newOutput['Date_of_Birth'] = recognizedOutput["dob_dd"] + '-' + recognizedOutput['dob_mm'] + '-' + recognizedOutput['dob_yyyy']
	newOutput['Date_of_Travel'] = recognizedOutput["dot_dd"] + '-' + recognizedOutput['dot_mm'] + '-' + recognizedOutput['dot_yyyy']

	newOutput["Name"] = recognizedOutput['name']
	newOutput["Aadhar_Number"] = recognizedOutput['aadhar']
	newOutput["Mobile_Number"] = recognizedOutput['mobile']
	newOutput["Train_Number"] = recognizedOutput['train']
	newOutput["Class"] = recognizedOutput['class']
	newOutput["Start_Station"] = recognizedOutput['start_stn']
	newOutput["End_Station"] = recognizedOutput['end_stn']
	
	return newOutput


recognizedOutput = {'date_dd': "15"}
#correction(recognizedOutput)