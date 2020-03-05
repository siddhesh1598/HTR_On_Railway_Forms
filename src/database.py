import csv

def updateDatabase(datalist):

	filename = "database.csv"

	header = ('Date', 'Name', 'Date_of_Birth',
			'Mobile_Number', 'Aadhar_Number',
			'Train_Number', 'Class',
			'Start_Station', 'End_Station',
			'Date_of_Travel')
	
	data = []
	for i in header:
		data.append(datalist[i])


	with open (filename, "a", newline = "") as csvfile:

		ticket = csv.writer(csvfile)
		ticket.writerow(header)
		ticket.writerow(data)



