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


#reading from the config file
def read_from_config(config_file_name,signal_list,other_arguments):
	signal_read = 0

	fptr=open(config_file_name,'r+')
	for row in fptr:
		#print 'row = ',row
		if(signal_read == 1):
			line = row.strip().split(' ')
			if(line[0] == 'end'):
				signal_read = 0
				continue
			else:
				sig_name = line[0]
				sig_time = int(line[1])
				sig_value = int(line[2])
			temp_signal = []
			temp_signal.append(sig_name)
			temp_signal.append(sig_time)
			temp_signal.append(sig_value)
			print sig_name,sig_time,sig_value
			signal_list.append(temp_signal)
			continue
		line = [i.strip() for i in row.strip().split('=')] #   row.strip().split('=').strip()
		if( line[0] == 'm'):
			m = line[1]
			print 'm = ',m
			continue
		if( line[0] == 'k'):
			k = line[1]
			print 'k = ',k
			continue
		if( line[0] == 'tmax'):
			tmax = line[1]
			print 'tmax = ',tmax
			continue
		if( line[0] == 'tmin'):
			tmin = line[1]
			print 'tmin = ',tmin
			continue
		if( line[0] == 'dataset_file'):
			csv_filename = line[1]
			print 'dataset file = ', csv_filename
			continue
		if( line[0] == 'start'):
			signal_read = 1
			continue
		if( line[0] == 'target'):
			if(len(line) == 2):
				target = line[1]
			if(len(line) == 3):
				target = line[1]+'= '+line[2]
			if(len(line) == 4):
				target = line[1]+' == '+line[3]
			target_predicate = target.split('"')[1]
			print 'target predicate = ', target_predicate
			# target_sig_name = target.split(' ')[0].strip()
			# print 'target signal name is-',target_sig_name
			continue
	other_arguments.append(csv_filename)
	other_arguments.append(m)
	other_arguments.append(k)
	other_arguments.append(tmax)
	other_arguments.append(tmin)
	other_arguments.append(target_predicate)

	
def find_target_columns(target_pred,signal_list):
	target_sig_name = target_pred.split()[0]
	for entry in signal_list:
		if entry[0] == target_sig_name :
			#return in format column_value,column_time
			return entry[2],entry[1]
	print 'The provided target is not in signal list.. or is not in correct format variable_name<space>operator<space>constant'


#find interval arguments are column number of value,column number of time,predicate
# the function will return an interval where the value of predicate is true
def find_Interval(target_predicate):
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
		
def compute_interval_length(interval):
	interval_length = 0.0
	for i in interval:
		interval_length += i[1] - i[0]
	# print 'interval ',interval
	#print 'interval length',interval_length
	return interval_length
def complement_interval(interval):
	c_interval = []
	temp=[0.0,0.0]
	c_interval.append(temp)
	for i in interval:
		c_interval[-1][1] = i[0]
		temp = [i[1],0.0]
		c_interval.append(temp)
	if(c_interval[-1][0] == initial_trace_length):
		del c_interval[-1]
	else:
		c_interval[-1][1]= initial_trace_length
	return c_interval

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
		if(start1 >= start2 and start1 <= end2):
			overlap = 1
		if(start2 >= start1 and start2 <= end1):
			overlap = 1
		if(overlap == 1):
			interval3.append([max(start1,start2),min(end1,end2)])
		if(end1 <= end2):
			i+=1
		else:
			j+=1
	return interval3

#change function name
#IMP_POINT TO NOTE : The function gives the sum of error of predicate p and not p
# so you can directly find the gain. 
def find_error_for_predicate(IntervalList,interval):
	#interval = find_Interval(predicate)
	print 'called find error'

	interval_complement = complement_interval(interval)	
	length_of_predicate_true = compute_interval_length(interval)
	length_of_predicate_false = compute_interval_length(interval_complement)

	if(length_of_predicate_true == 0.0 or length_of_predicate_false == 0.0):
		print 'predcate length zero'
		return 1000

	interval_true = merge(IntervalList[0] , interval)
	interval_false = merge(IntervalList[0] , interval_complement )
	
	predicate_and_target_true_length = compute_interval_length(interval_true)
	predicate_and_target_false_length = compute_interval_length(interval_false)

	print 'length p is true = ',length_of_predicate_true ,'\n length p and target true = ',predicate_and_target_true_length
	print 'length p is false = ',length_of_predicate_false ,'\n length p and target false = ',predicate_and_target_false_length

	if ( length_of_predicate_true >= predicate_and_target_true_length and length_of_predicate_false >= predicate_and_target_false_length):
		print 'true length and false length are smaller than len of predicate'
	else:
		print 'something is fishy in length'

	mean_true = predicate_and_target_true_length / length_of_predicate_true
	mean_false = predicate_and_target_false_length / length_of_predicate_false

	error_true = 2 * mean_true * (1-mean_true)
	error_false = 2 * mean_false * (1- mean_false)
	error_with_split = error_true + error_false

	print 'mean true =',mean_true
	print 'mean false = ',mean_false
	print 'error true = ',error_true
	print 'error false = ',error_false
	print 'error wtih split = ',error_with_split
	
	return error_with_split
	
	# mean = 0.0
	# for i in interval:
	# 	mean += i[1] - i[0]
	# error = (1.0 *mean)/0.04
	# error = error * (1-error)
	# return error

def find_error_for_predicate_target(interval):
	#interval = find_Interval(predicate)
	mean = 0.0
	for i in interval:
		mean += i[1] - i[0]
	
	print 'numerator',mean,'denominator',initial_trace_length
	mean = (mean)/initial_trace_length
	print 'mean for target  = ',mean
	error = 2 * mean * (1-mean)
	return error

#The function stores the m best predicates on the basics of gain
def store_m_best_predicates(gain,predicate):
	if gain <= 0:
		return
	if((gain,predicate) in m_best_predicates):
		print 'predicate already added to priority list'
		return
	print 'new potential predicate'
	index_new_predicate = 0
	if(len(m_best_predicates) == 0):
		m_best_predicates.append((gain,predicate))
		return
	else:
		for index in range(0,len(m_best_predicates)):
			if(gain > m_best_predicates[index][0]):
				index_new_predicate = index
				break
	if(index_new_predicate != m):
		m_best_predicates.insert(index_new_predicate,(gain,predicate))
	if(len(m_best_predicates) > m):
		del m_best_predicates[-1]

def print_m_best_predicates():
	print 'index \t gain \t\t predicate'
	for i in range(0,len(m_best_predicates)):
		print i+1,'\t',m_best_predicates[i][0],'\t',m_best_predicates[i][1]
	


# this fnction returns the best value of predicate we can get..
# with constant between min and max value of that variable
def generate_predicate(i,IntervalList,predicateList,curr_error):
	col_val= 2*i
	col_time = 2*i -1
	min_val=1000
	max_val=-1000
	first_line=1
	
	#the following block find the minimum and maximum value of the signal variable reading from the file.
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
	print '\n\ncolummn signal',i,'max_val= ',max_val,'min_val = ',min_val
	if(min_val == max_val):
		return 0,"blank",[]

	#temperature already set in main function
	#setting temperature parameters
	# Tmax=20
	# Tmin=10
	T=Tmax
	const_val = (max_val + min_val )/2
	constraint = 'value >= '+str(const_val)
	init_pred = (col_val,col_time,constraint)
	init_pred_interval = find_Interval(init_pred)
	print 'predicate : ', init_pred,' interval ',init_pred_interval
	error= find_error_for_predicate(IntervalList,init_pred_interval)
	# gain is how much the error is reduced
	print 'error target = ',curr_error
	print 'error after split',error

	gain = curr_error - error
	print 'predicate ',init_pred,'gain ',gain

	store_m_best_predicates(gain,init_pred)
	while(T>Tmin):
		displacement = (T - Tmin) * 1.0/(Tmax - Tmin)
		const_val_left = const_val - (const_val - min_val) * displacement
		const_val_right = const_val + (max_val - const_val) * displacement
		print '\n const val = ',const_val,'displacement = ',displacement,'left ',const_val_left,'right ',const_val_right
		
		#finding error for left val
		constraint_left = 'value >= '+str(const_val_left)
		init_pred_left = (col_val,col_time,constraint_left)
		init_pred_interval_left = find_Interval(init_pred_left)
		error_left = find_error_for_predicate(IntervalList,init_pred_interval_left)
		gain_left = curr_error - error_left
		store_m_best_predicates(gain_left,init_pred_left)
		
		#finding error for right value
		constraint_right = 'value >= '+str(const_val_right)
		init_pred_right = (col_val,col_time,constraint_right)
		init_pred_interval_right = find_Interval(init_pred_right)
		error_right= find_error_for_predicate(IntervalList,init_pred_interval_right)
		gain_right = curr_error - error_right
		store_m_best_predicates(gain_right,init_pred_right)
		
		print 'gain = ',gain,'gain_left ',gain_left,'gain_right = ',gain_right
		

		if(gain_left > gain and gain_left >= gain_right):
			const_val = const_val_left
			gain = gain_left
			init_pred_interval = init_pred_interval_left
		elif(gain_right > gain and gain_right >=gain_left):
			const_val = const_val_right
			gain = gain_right
			init_pred_interval = init_pred_interval_right
		elif(gain > gain_right and gain > gain_left):
			#if new solution is not good accept it with some probability
			if(gain_left >= gain_right):
				if( displacement > random.random()):
					const_val = const_val_left
					gain = gain_left
					init_pred_interval = init_pred_interval_left
			else:
				if( displacement > random.random()):
					const_val = const_val_right
					gain = gain_right
					init_pred_interval = init_pred_interval_right




		#decrease temperature
		T-=1

	constraint = 'value >= '+str(const_val)
	init_pred = (col_val,col_time,constraint)
	
	return gain,init_pred,init_pred_interval
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
	#local variables 
	max_gain = 0.0
	best_predicate = (-1,-1,'val >= 0')
	best_predicate_interval = []
	any_variable_to_add = 0

	#for every signal variable check best predicate that could be generated.
	for i in addedvariable:
		# print 'value of i is',if
		if(addedvariable[i] == 0):
			#here we are calling function that returns us best predicate for i th signal variable
			gain,predicate,interval = generate_predicate(i,IntervalList,predicateList,curr_error)
			any_variable_to_add = 1
			if(gain == 0):
				print 'got gain 0 for ',i
				addedvariable[i] = -1
			else:
				if(gain > max_gain):
					max_gain = gain
					best_predicate = predicate
					best_predicate_interval = interval

	if( any_variable_to_add == 0 or max_gain == 0):
		dummy_predicate = (-1,-1,'val >= 0')
		return dummy_predicate			
	
	# mxgain_index=0
	# for ind in range(0,len(gainlist)):
	# 	if(gainlist[ind][0] > gainlist[mxgain_index][0]):
	# 		mxgain_index = ind
	# return gainlist[mxgain_index][1]

	return best_predicate,best_predicate_interval


#main function hard coded the target signal value
if __name__ == "__main__":
		arguments = sys.argv
		if(len(arguments) == 1):
			print 'Missing  arguments please give arguments as follows'
			print 'arguments:-  1)file name, 2)target column for time,3) target column for value 4) value of m(best predicates to remember)'

		signal_list = []
		other_arguments = []
		#find interval arguments are column number of value,column number of time,predicate
		addedvariable = {}	#the variables of signal that have been added as predicates
		for i in range(1,15):
			addedvariable[i]=0
		config_file_name = 'config_file_unconstrained.txt'
		print 'calling read_config'
		read_from_config(config_file_name,signal_list,other_arguments)
		print 'other arguments ',other_arguments
		# target_time = 19
		# target_col = 20
		# pred = 'value >= 2'
		# addedvariable[target_col/2] = 1
		# m=5
		global csv_filemame
		csv_filename = other_arguments[0]
		
		global trace_length
		trace_length = float(other_arguments[1])

		global initial_trace_length
		initial_trace_length = trace_length

		k= int(other_arguments[2])

		# target_sig_name = other_arguments[3]
		#making Tmax and Tmin global so they can be accessed easily
		global Tmax
		global Tmin
		
		Tmax = int(other_arguments[3])
		Tmin = int(other_arguments[4])
		target_pred = other_arguments[5]
		target_col,target_time = find_target_columns(target_pred,signal_list)
		print 'target columns are',target_time,target_col
		#m is declared global so it can be easily accessed throughout the code and w dont need to pass it.
		global m 
		m = trace_length
		global m_best_predicates
		m_best_predicates = []
		
		

		# print 'Please Enter the predicate'
		# pred=raw_input()
		# print 'Please Enter the value of max temperature'
		# Tmax=int(raw_input())
		# print 'Please Enter the value of min temperature'
		# Tmin=int(raw_input())

		

		addedvariable[target_col/2] = 1
		print '********************************************************'
		print 'THE ARGUMENTS ARE AS FOLLOWS'
		print '1. The file name of csv file is: ',csv_filename
		print '2. target predicate time column:',target_time
		print '3. target predicate value column:',target_col
		print '4. predicate: ',target_pred
		print '5. value of m:',m
		print '********************************************************\n\n'
		
		#a predicate structure consisting of column number of value and time in time series data and
		#predicate of the form value >= constant
		target_predicate =(target_col,target_time,target_pred)
		interval=find_Interval(target_predicate)
		interval_length_target=compute_interval_length (interval)
		IntervalList.append(interval)
		predicateList.append(target_predicate)
		curr_error = find_error_for_predicate_target(interval)
		print 'target curr error is',curr_error,'\n'
		
		number_of_predicate = 5
		loop_counter = 1
		while(len(predicateList) < number_of_predicate):
			print 'Calling simulated annealing for iteration',loop_counter
			new_predicate,new_predicate_interval = sa(IntervalList,predicateList,addedvariable,curr_error)
			
			#when there is no new pedicate
			if(new_predicate[0] == -1):
				break

			print 'choosen predicate in iteration ',loop_counter,'is ',new_predicate
			predicateList.append(new_predicate)
			variable = new_predicate[0]/2
			addedvariable[variable] = 1
			IntervalList[0]=merge(IntervalList[0],new_predicate_interval)
			trace_length = compute_interval_length(IntervalList[0])
			loop_counter+=1

		
		for i in predicateList:
			print i
		print 'The ', m,' best predicates found thoughout program run in decreasing order of gain:\n'
		print_m_best_predicates()