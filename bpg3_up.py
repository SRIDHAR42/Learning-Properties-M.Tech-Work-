 #################################################################
 #                      CODE WRITTEN BY                          
 #                                                               
 #                  Lal Sridhar Vaishnava                  			
 #        M.Tech 2nd Year, Dept. of CSE, IIT Kharagpur     
 #                                                               
 #################################################################
import csv
import os
import random
import sys

# data structures
# 1 Tuple: Tuple for storing predicate in the form(col,op,const value)
# 	the constant value will be between min and max of the value of signal



def disp():
	i=1;
	for row in filereader:
		print row
		i+=1
		if (i == 10):
			break

predicateList=[]
IntervalList=[]

#find interval arguments are column number of value,column number of time,predicate
# the function will return an interval where the value of predicate is true
def findInterval(target_predicate):
	col_val,col_time,target = target_predicate
	f=open("test1",'w+')
	f.truncate()
	target_val=target.split()
	f.write("start 2\n\n")
	f.write(target_val[0]+" "+ str(col_val) + ' ' + str(col_time))
	f.write('\n\nbegin\n\n')
	f.write(target)
	f.write('\n\nend\n')
	f.close()
	booleanize_command = './Booleanize test1 ' + csv_filename + ' > uselessfile'
	#os.system('./Booleanize test1 LDO_netlist_dump.csv > uselessfile')
	os.system(booleanize_command)
	f=open('list.dat','r+')
	interval=[]
	line=''
	for row in f:
		line=row
	f.close()
	if(line == '{}'):
		return interval
	firstline=True
	for i in line.split(')')[:-1]:
		if(firstline):
			line= i[2:]
			firstline=False
		else:
			line= i[1:]
		line1=line.split(':')
		first,second=float(line1[0]),float(line1[1])
		interval.append([first,second])
	return interval
		
		



def func(target):
	list_target=findInterval(target)
	predicateList.append(list_target)


def merge(interval1,interval2):
	interval3=[]
	len1=len(interval1)
	len2=len(interval2)
	i=0
	j=0
	while(i<len1 and j<len2):
		start1,end1=interval1[i]
		start2,end2=interval2[j]
		overlap = 0
		if(start1 > start2 and start1 < end2):
			overlap = 1
		if(start2 > start1 and start2 < end1):
			overlap = 1
		if(overlap == 1):
			interval3.append([max(start1,start2),min(end1,end2)])
		if(end1 < end2):
			i+=1
		else:
			j+=1
	return interval3

#change function name
def find_error_for_predicate(predicate,IntervalList,predicateList):
	interval = findInterval(predicate)
	interval = merge(IntervalList[0] , interval)
	mean = 0.0
	for i in interval:
		mean += i[1] - i[0]
	error = (1.0 *mean)/0.04
	error = error * (1-error)
	return error

def find_error_for_predicate_target(predicate):
	interval = findInterval(predicate)
	mean = 0.0
	for i in interval:
		mean += i[1] - i[0]
	print 'mean = ',mean
	error = (1.0 *mean)/0.04
	error = error * (1-error)
	return error


# this fnction returns the best value of predicate we can get..
# with constant between min and max value of that variable
def generate_predicate(i,IntervalList,predicateList,curr_error):
	col_val= 2*i
	col_time = 2*i -1
	min_val=1000
	max_val=-1000
	first_line=1
	
	#the following block find the minimum and maximum value of the signal variable
	with open(csv_filename,'rb') as csvfile:
		filereader=csv.reader(csvfile,delimiter=',')
		for row in filereader:
			if(first_line == 1):
				#first line is the signal variable name so we skip that.
				first_line = 0
				continue
			val=float(row[col_val-1])
			min_val = min(min_val,val)
			max_val = max(max_val,val)
	print 'colummn signal',i,'max_val= ',max_val,'min_val = ',min_val
	if(min_val == max_val):
		return 0,"blank"

	
	#setting temperature parameters
	Tmax=20
	Tmin=10
	T=Tmax
	const_val = (max_val - min_val )/2
	constraint = 'value >= '+str(const_val)
	init_pred = (col_val,col_time,constraint)
	error = find_error_for_predicate(init_pred,IntervalList,predicateList)
	# gain is how much the error is reduced
	gain = curr_error - error
	while(T>Tmin):
		displacement = (T - Tmin) * 1.0/(Tmax - Tmin)
		const_val_left = const_val - (const_val - min_val) * displacement
		const_val_right = const_val + (max_val - const_val) * displacement
		print 'const val = ',const_val,'displacement = ',displacement,'left ',const_val_left,'right ',const_val_right
		
		#finding error for left val
		constraint_left = 'value >= '+str(const_val_left)
		init_pred_left = (col_val,col_time,constraint_left)
		error_left = find_error_for_predicate(init_pred_left,IntervalList,predicateList)
		gain_left = curr_error - error_left
		
		#finding error for right value
		constraint_right = 'value >= '+str(const_val_right)
		init_pred_right = (col_val,col_time,constraint_right)
		error_right = find_error_for_predicate(init_pred_right,IntervalList,predicateList)
		gain_right = curr_error - error_right
		
		print 'gain = ',gain,'gain_left ',gain_left,'gain_right = ',gain_right
		

		if(gain_left > gain and gain_left >= gain_right):
			const_val = const_val_left
			gain = gain_left
		elif(gain_right > gain and gain_right >=gain_left):
			const_val = const_val_right
			gain = gain_right
		elif(gain > gain_right and gain > gain_left):
			#if new solution is not good accept it with some probability
			if(gain_left >= gain_right):
				if( displacement > random.random()):
					const_val = const_val_left
					gain = gain_left
			else:
				if( displacement > random.random()):
					const_val = const_val_right
					gain = gain_right




		#decrease temperature
		T-=1

	constraint = 'value >= '+str(const_val)
	init_pred = (col_val,col_time,constraint)
	error = find_error_for_predicate(init_pred,IntervalList,predicateList)
	gain = curr_error - error
	
	return gain,init_pred
	# const_val = min_val
	# mx_mean = 0
	# mx_predicate = (-1,-1,'')
	# while( const_val < max_val):
	# 	print 'const value',const_val
	# 	constraint = 'y >= '+str(const_val)
	# 	init_pred = (col_val,col_time,constraint)
	# 	error = find_mean_for_predicate(init_pred,IntervalList,predicateList)
	# 	gain = error - curr_error
	# 	#gain = curr_error - error
	# 	print 'signal variable ',i,' gain ',gain
	# 	if(gain > mx_mean):
	# 		mx_predicate = init_pred
	# 		mx_mean=gain

	# 	const_val += 0.15
	# return mx_mean,mx_predicate
	



def sa(IntervalList,predicateList,addedvariable,curr_error):
	gainlist=[]
	any_variable_to_add = 0
	for i in addedvariable:
		if(addedvariable[i] == 0):
			#here we are calling function that returns us best predicate for i th signal variable
			gain,predicate = generate_predicate(i,IntervalList,predicateList,curr_error)
			any_variable_to_add = 1
			if(gain == 0):
				print 'got gain 0 for ',i
				addedvariable[i] = -1
			else:
				gainlist.append([gain,predicate])

	if( any_variable_to_add == 0):
		dummy_predicatte = (-1,-1,'val >= 0')
		return dummy_predicatte			
	print 'printing predicates \n\n'
	for item in gainlist:
		print item
	print 'printing predicates ended \n\n'
	
	mxgain_index=0

	for ind in range(0,len(gainlist)):
		if(gainlist[ind][0] > gainlist[mxgain_index][0]):
			mxgain_index = ind
	return gainlist[mxgain_index][1]


#main function hard coded the target signal value
if __name__ == "__main__":
		
		arguments = sys.argv
		if(len(arguments) == 1):
			print 'Missing  arguments please give arguments as follows'
			print 'arguments:-  1)file name, 2)target column for time,3) target column for value'


		#find interval arguments are column number of value,column number of time,predicate
		addedvariable = {}	#the variables of signal that have been added as predicates
		for i in range(1,15):
			addedvariable[i]=0
		
		#setting target parameters manual in comment and command line
		
		# target_time = 19
		# target_col = 20
		# pred = 'value >= 2'
		# addedvariable[target_col/2] = 1
		csv_filename = arguments[1]
		global csv_filemame
		target_time = int(arguments[2])
		target_col = int(arguments[3])
		print 'Please Enter the predicate \n'
		pred=raw_input()
		addedvariable[target_col/2] = 1
		print '********************************************************'
		print 'THE ARGUMENTS ARE AS FOLLOWS'
		print '1. The file name of csv file is: ',csv_filename
		print '2. target predicate time column:',target_time
		print '3. target predicate value column:',target_col
		print '4. predicate: ',pred
		print '********************************************************\n\n'
		
		#a predicate structure consisting of column number of value and time in time series data and
		#predicate of the form value >= constant
		target_predicate =(target_col,target_time,pred)
		interval=findInterval(target_predicate)
		IntervalList.append(interval)
		predicateList.append(target_predicate)
		curr_error = find_error_for_predicate_target(target_predicate)
		print 'curr error is',curr_error
		
		number_of_predicate = 5
		loop_counter = 1
		while(len(predicateList) < number_of_predicate):
			print 'Calling simulated annealing for iteration',loop_counter
			new_predicate = sa(IntervalList,predicateList,addedvariable,curr_error)
			
			#when there is no new pedicate
			if(new_predicate[0] == -1):
				break

			print 'choosen predicate in iteration ',loop_counter,'is ',new_predicate
			predicateList.append(new_predicate)
			variable = new_predicate[0]/2
			addedvariable[variable] = 1
			IntervalList[0]=merge(IntervalList[0],findInterval(new_predicate))
			loop_counter+=1

		
		for i in predicateList:
			print i