 #################################################################
 #                      CODE WRITTEN BY                          
 #                                                               
 #                  Lal Sridhar Vaishnava                  			
 #        M.Tech 2nd Year, Dept. of CSE, IIT Kharagpur     
 #                                                               
 #################################################################

# UPDATES TO MAKE IN THE CODE

# if gain is same then
# for >= store biggest constant
# for <= store minimum constant





import csv
import os
import random
import sys
# from pathlib import Path

# data structures
# 1 Tuple: Tuple for storing predicate in the form(col,op,const value)
# 	the constant value will be between min and max of the value of signal



# DATA STRUCTURE DESCRIPTION
# predicate is a tuple (signalname,operator,constant)
# m_best_predicates is a list where each element is (gain,predicate,bucket_number)

def print_dictionary(temp_dict):
	for i in temp_dict:
		print i,temp_dict[i]

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
def read_from_config(config_file_name,signal_list,other_arguments,influence_list):
	signal_read = 0
	influence_read = 0
	target_read = 0
	trace_length = 0
	fptr=open(config_file_name,'r+')
	for row in fptr:
		#print 'row = ',row
		if row == '\n':
			continue
		# Reading List of Signals
		if(signal_read == 1 and row != '\n'):
			line = row.strip().split(' ')
			if(line[0] == 'end'):
				signal_read = 0
				continue
			else:
				sig_name = line[0]
				sig_time = int(line[2])
				sig_value = int(line[1])
			temp_signal = []
			temp_signal.append(sig_name)
			temp_signal.append(sig_time)
			temp_signal.append(sig_value)
			print sig_name,sig_time,sig_value
			signal_list.append(temp_signal)
			continue
		# Reading the Influence List
		if(influence_read == 1 and row != '\n'):
			line = row.strip().split(' ')
			if(line[0] == 'influence_list_end'):
				influence_read = 0
				print 'reading influence list ended'
				continue
			else:
				line = row.split(' ')
				bucket_number = int(line[0])
				# print line[1]
				bucket_interval = parse_interval_from_string(line[1],trace_length)
				current_influence_list_entry = []
				current_influence_list_entry.append(bucket_number)
				current_influence_list_entry.append(bucket_interval)
				influence_list.append(current_influence_list_entry)
				# print current_influence_list_entry
			continue

		# Reading target interval
		if(target_read == 1 and row != '\n'):
			line = row.strip().split(' ')
			if(line[0] == 'target_end'):
				target_read = 0
				continue
			else:
				target_interval = parse_interval_from_string(row.strip(),trace_length)
			continue

		line = [i.strip() for i in row.strip().split('=')] #   row.strip().split('=').strip()
		if( line[0] == 'COMMENTS'):
			print 'found comments config file reading ended'
			break
		if( line[0] == 'm'):
			m = line[1]
			print 'm = ',m
			continue
		if( line[0] == 'trace_length'):
			trace_length = float(line[1])
			print 'trace length = ',trace_length
			continue
		if( line[0] == 'k'):
			k = line[1]
			print 'k = ',k
			continue
		if( line[0] == 'n'):
			n = line[1]
			print 'n = ',n
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
		if( line[0] == 'influence_list_start'):
			influence_read = 1
			print 'reading influence list'
			continue
		if( line[0] == 'target_start'):
			target_read = 1
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
	other_arguments.append(n)
	other_arguments.append(trace_length)
	other_arguments.append(target_interval)

def find_min_max_of_variable(i):
	for entry in minmax_of_signal_variables:
		if entry[0] == i:
			# return the minimum and maximum value
			return entry[1],entry[2]
	print 'Something wrong the signal variable does not exist'

def find_column(sig_name):
	for entry in signal_list:
		if entry[0] == sig_name :
			#return in format column_value,column_time
			return entry[2],entry[1]
	
def find_target_columns(target_pred,signal_list):
	target_sig_name = target_pred.split()[0]
	for entry in signal_list:
		if entry[0] == target_sig_name :
			#return in format column_value,column_time
			return entry[2],entry[1]
	print 'The provided target is not in signal list.. or is not in correct format variable_name<space>operator<space>constant'

def compute_min_max_of_signal_variable(csv_filename,signal_list ,minmax_of_signal_variables):
	minmax_file_name = 'minimum_maximum_' + csv_filename[:-4] +'.txt'
	print 'minmax file name is ',minmax_file_name
	my_file = os.path.exists('./'+minmax_file_name)
	if my_file:
		filereader = open(minmax_file_name,'r+')
		for row in filereader:
				if(row == '\n'):
					continue
				line = row.strip().split(',')
				# print 'line ',line
				sig_name,min_val,max_val = line
				min_val = float(min_val)
				max_val = float(max_val)
				minmax_entry = []
				minmax_entry.append(sig_name)
				minmax_entry.append(min_val)
				minmax_entry.append(max_val)
				minmax_of_signal_variables.append(minmax_entry)
		print 'read minmax values from file'
		return
	else:
		print 'file not present'
	print 'computing min and max for every signal variable'
	fptr = open(minmax_file_name,'w+')
	fptr.truncate()

	for entry in signal_list:
		col_val = entry[2]
		min_val=1000
		max_val=-1000
		first_line = 1
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
		minmax_entry = []
		minmax_entry.append(entry[0])
		minmax_entry.append(min_val)
		minmax_entry.append(max_val)
		fptr.write(entry[0] + ',' +str(min_val)  + ',' +str(max_val) + '\n')
		minmax_of_signal_variables.append(minmax_entry)
	print 'computed min and max for every signal variable'

def eleminate_useless_signals(processed_signal_variable , minmax_of_signal_variables):
	for entry in minmax_of_signal_variables:
		# comparing minimum and maximum values
		if entry[1] == entry[2] :
			# that signal variable is marked as processed
			processed_signal_variable[entry[0]] = 1
			print 'The signal variable ',entry[0],'has same minmax value and hence is marked as already processed'
	print 'Eliminating useless signal completed'


#find interval arguments are column number of value,column number of time,predicate
# the function will return an interval where the value of predicate is true
def find_Interval(target_predicate):
	col_val,col_time,target = target_predicate
	# print 'finding interval for ',target_predicate
	f=open("test1",'w+')
	f.truncate()
	target_val=target.split()
	f.write("start 1\n\n")
	f.write(target_val[0]+" "+ str(col_val) + ' ' + str(col_time))
	f.write('\n\nbegin\n\n')
	f.write(target)
	f.write('\n\nend\n')
	f.write('(1)')
	f.close()
	booleanize_command = './Booleanize test1 ' + csv_filename + ' > uselessfile'
	#os.system('./Booleanize test1 LDO_netlist_dump.csv > uselessfile')
	os.system(booleanize_command)
	output_filename = csv_filename.split('.')[0] + '.dat'
	# print 'output filename is ',output_filename
	f=open(output_filename,'r+')
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
		if(second >= initial_trace_length):
			second = initial_trace_length
			stop_computing = 1
		if(first < initial_trace_length):
			interval.append([first,second])
	return interval
		
def parse_interval_from_string(line,initial_trace_length):
	# print 'line for config',line
	interval=[]
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
		first,second=float(line1[0]) ,float(line1[1])
		# print 'first',first,'second',second
		# print 'second is ',second,'initial_trace_length is ',initial_trace_length
		if( abs(initial_trace_length - second ) <= 0.000001):
			# print 'found true'
			second = initial_trace_length
			stop_computing = 1
		if(first < initial_trace_length):
			temp = []
			temp.append(first)
			temp.append(second)
			interval.append(temp)
			# interval.append([first,second])
	# print 'interval extracted ',interval
	# print 'initial trace_length',initial_trace_length
	return interval

def generate_pseudo_targets(pseudo_targets,n,k):
	i=0
	
	while(i <= n):
		pseudo_targets[-1 * i] = minkowski_difference(target_interval,i*k) 
		pseudo_targets_error[-1 * i] = find_error_for_predicate_target( pseudo_targets[-1 * i])
		print 'pseudo_targets error for (-',i,')',pseudo_targets_error[-1 *i]
		i+=1

def generate_pseudo_complement_targets(pseudo_complement_targets,n,k):
	i=0
	while(i <= n):
		pseudo_complement_targets[-1 * i] = minkowski_difference(target_false_interval,i*k) 
		pseudo_complement_targets_error [ -1 * i] = find_error_for_predicate_target (pseudo_complement_targets[-1 * i])
		print 'pseudo_complement_targets_errorfor (-',i,')',pseudo_complement_targets_error [ -1 * i]
		i+=1


def merge_overlapping_intervals(interval):
	n_interval = []
	if(len(interval) == 0):
		return n_interval
	n_interval.append(interval[0])
	i=1
	index = 0
	interval_length = len(interval)
	while(i < interval_length):
		if(interval[i][0] <= n_interval[index][1] ):
			n_interval[index][1] = interval[i][1]
		else:
			n_interval.append(interval[i])
			index+=1
		i+=1
	return n_interval

def minkowski_sum(interval,k):
	n_interval = []
	stop_computing = 0
	for entry in interval:
		temp = []
		temp.append(entry[0])
		end = entry[1] + float(k)
		if(end >= initial_trace_length):
			end = initial_trace_length
			stop_computing = 1
		temp.append(end)
		# temp.append(min(initial_trace_length ,entry[1] + float(k)))
		n_interval.append(temp)
		if (stop_computing == 1):
			break
	n_interval = merge_overlapping_intervals(n_interval)
	return n_interval


# doubt to ask from antonio sir wheather trace starts from 0
def minkowski_difference(interval,k):
	# print 'interval is',interval
	n_interval = []
	stop_computing = 0
	for entry in reversed(interval) :
		temp = []
		start = entry[0] - float(k)
		if( start < 0  ):
			temp.append(0)
			temp.append(entry[1])
			stop_computing = 1
		else:
			temp.append(start)
			temp.append(entry[1])
		n_interval.append(temp)
		if(stop_computing == 1):
			break
	
	n_interval.reverse()

	# print 'before ',interval
	# print 'after minkowski diff ',n_interval
	n_interval = merge_overlapping_intervals(n_interval)
	return n_interval

def sort_influence_list(influence_list):
	list_len = len(influence_list)
	if(list_len < 2):
		return
	i=0
	j=1
	while(i<list_len):
		j=i+1
		while(j<list_len):
			# print i,j
			if influence_list[i][0] < influence_list[j][0] :
				influence_list[i],influence_list[j] = influence_list[j],influence_list[i]
			j+=1
		i+=1

def fill_intersected_influence_list(n):
	for i in range(n+1):
		intersected_influence_list[i] = []
	for entry in influence_list:
		bucket_number,bucket_interval=entry
		intersected_influence_list[bucket_number] = bucket_interval
	print 'printing intersected_influence_list'

	i = n-1
	while(i >= 0):
		if len(intersected_influence_list[i + 1])!=0:
			temp_interval = minkowski_sum(intersected_influence_list[i + 1],k)
			if(len(intersected_influence_list[i]) == 0):  
				intersected_influence_list[i]= temp_interval
			else:
				intersected_influence_list[i] = intersect_interval(temp_interval,intersected_influence_list[i])
		i-=1
	print_dictionary(intersected_influence_list)


# The following function intersect the predicate with the influence list keeping track of the resolution
# and returns an interval that is couse part of final assertion

# change name copute forward influence list
def compute_forward_influence_list(predicate_interval,bucket_number):
	temp_minimum_bucket_value = min(bucket_number,initial_minimum_bucket_value)
	# print 'bucket number',bucket_number,'temp_minimum_bucket_value',temp_minimum_bucket_value
	i = bucket_number
	if(intersected_influence_list[i] == []):
		temp_influence_entry_interval = predicate_interval
	else:
		# print 'intersecting with ',i
		temp_influence_entry_interval = intersect_interval(predicate_interval,intersected_influence_list[i])
	i-=1
	while(i >= temp_minimum_bucket_value):
		# print 'intersecting with ',i
		temp_influence_entry_interval = minkowski_sum(temp_influence_entry_interval,k)
		if(intersected_influence_list[i] == []):
			temp_influence_entry_interval = predicate_interval
		else:
			temp_influence_entry_interval = intersect_interval(predicate_interval,intersected_influence_list[i])
		i -= 1
	# print 'predicate interval',predicate_interval
	# print 'bucket_number',bucket_number
	# print temp_influence_entry_interval
	# print 'returning from compute_forward_influence_list'
	return temp_influence_entry_interval


def get_min_bucket_value(influence_list):
	mbv = n
	for entry in influence_list:
		if (entry[0] < mbv):
			mbv = entry[0]
	return mbv

def compute_interval_length(interval):
	interval_length = 0.0
	for i in interval:
		interval_length += i[1] - i[0]
	# print 'interval ',interval
	#print 'interval length',interval_length
	return interval_length
def complement_interval(interval):
	c_interval = []
	if(len(interval) == 0):
		temp = []
		temp.append(0)
		temp.append(initial_trace_length)
		c_interval.append(temp)
		return c_interval
	# print 'finding complement for ',interval
	
	temp=[0.0,0.0]
	c_interval.append(temp)
	for i in interval:
		c_interval[-1][1] = i[0]
		temp = [i[1],0.0]
		c_interval.append(temp)
	if(c_interval[-1][0] > initial_trace_length):
		print 'anaoly found time value bigger than initial trace length'
	if(c_interval[-1][0] == initial_trace_length):
		del c_interval[-1]
	else:
		c_interval[-1][1]= initial_trace_length

	if(interval[0][0] == 0):
		del c_interval[0]
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

def intersect_interval(interval1,interval2):
	interval3=[]
	# print 'interval 1',interval1
	# print 'interval 2',interval2
	len1=len(interval1)
	len2=len(interval2)
	# if(len1 == 0 or len2 == 0):
	# 	return interval3
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
	# print 'interval 3',interval3
	return interval3


def compute_mean_two_interval(predicate_interval , target_interval):
	# print 'predicate_interval',predicate_interval
	# print 'target_interval',target_interval
	intersected_interval = intersect_interval(predicate_interval,target_interval)
	# print 'intersected_interval',intersected_interval
	len_intersected_interval = compute_interval_length(intersected_interval)
	# print 'len_intersected_interval',len_intersected_interval
	len_predicate_interval = compute_interval_length(predicate_interval)
	# print 'len_predicate_interval',len_predicate_interval
	if(len_predicate_interval == 0):
		# print 'found empty interval while computing mean'
		return -1
	mean_of_intervals = len_intersected_interval / len_predicate_interval
	return mean_of_intervals


#change function name
#IMP_POINT TO NOTE : The function gives the sum of error of predicate p and negation p
# so you can directly find the gain. 
def find_error_for_predicate(predicate,bucket_number):
	#interval = find_Interval(predicate)
	temp_minimum_bucket_value = min(bucket_number,initial_minimum_bucket_value)
	predicate_interval = find_Interval(predicate)
	complement_predicate_interval = complement_interval(predicate_interval)

	processed_predicate_interval = compute_forward_influence_list(predicate_interval,bucket_number)
	
	processed_complemant_predicate_interval = compute_forward_influence_list(complement_predicate_interval,bucket_number)

	choosen_pseudo_target = pseudo_targets[-1 * temp_minimum_bucket_value]

	choosen_pseudo_complement_target = pseudo_complement_targets[-1 * temp_minimum_bucket_value]


	print 'printing various intervals'
	print 'predicate interval'
	print predicate_interval
	print 'complement_predicate_interval'
	print complement_predicate_interval
	print 'processed predicate interval'
	print processed_predicate_interval
	print 'processed complement predicate interval'
	print processed_complemant_predicate_interval
	print 'choosen target and pseudo complement target'
	print choosen_pseudo_target
	print choosen_pseudo_complement_target


	mean_for_predicate = compute_mean_two_interval(processed_predicate_interval,choosen_pseudo_target)
	mean_for_predicate_false = compute_mean_two_interval(processed_complemant_predicate_interval,choosen_pseudo_target)

	false_mean_for_predicate = compute_mean_two_interval(processed_predicate_interval,choosen_pseudo_complement_target)
	false_mean_for_predicate_false = compute_mean_two_interval(processed_complemant_predicate_interval,choosen_pseudo_complement_target)

	print 'mean_for_predicate'
	print mean_for_predicate
	print 'mean for predicate false'
	print mean_for_predicate_false
	print 'false mean for predicate'
	print false_mean_for_predicate
	print 'false mean for predicate false'
	print false_mean_for_predicate_false

	# if any of the mean is -1 the it means it is bad and should not be taken
	if (mean_for_predicate == -1 ):
		true_error_predicate = 1000
	else:
		true_error_predicate = 2 * mean_for_predicate * (1 - mean_for_predicate )
	

	if (mean_for_predicate_false == -1 ):
		true_error_predicate_false = 1000
	else:
		true_error_predicate_false = 2 * mean_for_predicate_false * (1- mean_for_predicate_false )


	if ( false_mean_for_predicate == -1 ):
		false_error_predicate = 1000
	else:
		false_error_predicate = 2 * false_mean_for_predicate * ( 1- false_mean_for_predicate )

	
	if (false_mean_for_predicate_false == -1 ):
		false_error_predicate_fasle = 1000
	else:
		false_error_predicate_fasle = 2 * false_mean_for_predicate_false * (1 - false_mean_for_predicate_false)

	extra_info = []
	if(true_error_predicate < false_error_predicate):
		extra_info.append('left branch true error')
		extra_info.append(true_error_predicate)
	else:
		extra_info.append('left branch false error')
		extra_info.append(false_error_predicate)
	if(true_error_predicate_false < false_error_predicate_fasle):
		extra_info.append('right branch true error')
		extra_info.append(true_error_predicate_false)
	else:
		extra_info.append('right branch false error')
		extra_info.append(false_error_predicate_fasle)

	error_predicate = min(true_error_predicate,false_error_predicate)
	error_predicate_false = min(true_error_predicate_false,false_error_predicate_fasle)

	error_with_split = error_predicate + error_predicate_false 

	return error_with_split,extra_info

	
def find_error_for_predicate_target(interval):
	#interval = find_Interval(predicate)
	mean = 0.0
	for i in interval:
		mean += i[1] - i[0]
	mean = (mean)/initial_trace_length
	# print 'mean for target  = ',mean
	if(mean > 1 or mean < 0):
		print 'anomaly found in mean for interval ',interval
	error = 2 * mean * (1-mean)
	return error



#The function stores the m best predicates on the basics of gain
def store_m_best_predicates(gain,predicate,bucket_number,extra_info):
	if gain <= 0:
		return
	
	for i in range(len(m_best_predicates)):
		item_gain,item_predicate,item_bucket_number,item_extra_info = m_best_predicates[i]
		print 'both gains are ',item_gain,gain
		# if(item_gain == gain):
		# 	print 'they are equal'
		# else:
		# 	print 'they are not equal'
		if(item_gain == gain and item_bucket_number == bucket_number):
			item_predicate_signalname,item_predicate_operator,item_predicate_constant = item_predicate
			predicate_signalname,predicate_operator,predicate_constant = predicate
			
			if(item_predicate_signalname == predicate_signalname and item_predicate_operator == predicate_operator):
				if(predicate_operator == '>=' and predicate_constant >= item_predicate_constant):
					m_best_predicates[i][1] = predicate
					m_best_predicates[i][3] = extra_info
					return
				if(predicate_operator == '<=' and predicate_constant <= item_predicate_constant):
					m_best_predicates[i][1] = predicate
					m_best_predicates[i][3] = extra_info
					return
				return
		
		
	if((gain,predicate,bucket_number,extra_info) in m_best_predicates):
		print 'predicate already added to priority list'
		return

	index_new_predicate = 0
	len_m_best_predicates = len(m_best_predicates)
	if(len_m_best_predicates == 0):
		m_best_predicates.append((gain,predicate,bucket_number,extra_info))
		return
	else:
		while ( index_new_predicate < len_m_best_predicates):
			if(gain > m_best_predicates[index_new_predicate][0]):
				break
			index_new_predicate+=1
	# if(index_new_predicate != m):
	m_best_predicates.insert(index_new_predicate,(gain,predicate,bucket_number,extra_info))
	if(len(m_best_predicates) > m):
		del m_best_predicates[-1]

def print_m_best_predicates():
	print 'index \t gain \t\t predicate \t\t bucket_number \t left branch info \t right branch info',
	for i in range(0,len(m_best_predicates)):
		print i+1,'\t',m_best_predicates[i][0],'\t',m_best_predicates[i][1],m_best_predicates[i][2],m_best_predicates[i][3]
	
# this fnction returns the best value of predicate we can get for a given operator op..
# with constant between min and max value of that variable
def  generate_predicate(i,op, bucket_number ,curr_error):
	col_val,col_time = find_column(i)
	min_val , max_val = find_min_max_of_variable(i)
	store_error = {}
	
	print '\n\n signal',i,'max_val= ',max_val,'min_val = ',min_val,'bucket number =',bucket_number
	if(min_val == max_val):
		return 0,"blank",[]

	#temperature already set in main function
	#setting temperature parameters
	# Tmax=20
	# Tmin=10
	T=Tmax
	const_val = (max_val + min_val )/2
	constraint = i + ' ' + op + ' ' + str(const_val)
	# constraint = 'value >= '+str(const_val)

	init_pred = (col_val,col_time,constraint)
	# init_pred_interval = find_Interval(init_pred)
	print 'predicate : ', init_pred
	error,extra_info = find_error_for_predicate(init_pred, bucket_number)
	store_error[const_val] = error
	# gain is how much the error is reduced
	
	print 'error target = ',curr_error
	print 'error after split',error

	gain = curr_error - error
	print 'predicate ',init_pred,'gain ',gain

	store_m_best_predicates(gain,init_pred,bucket_number,extra_info)

	while(T>Tmin):
		displacement = (T - Tmin) * 1.0/(Tmax - Tmin)
		const_val_left = const_val - ((const_val - min_val) * displacement)
		const_val_right = const_val + ((max_val - const_val) * displacement)
		print '\n const val = ',const_val,'displacement = ',displacement,'const_val_left ',const_val_left,'const_val_right ',const_val_right
		
		#finding error for left val
		if const_val_left in store_error:
			error_left = store_error[const_val_left]
			gain_left = curr_error - error_left
		else:
			constraint_left = i + ' ' + op + ' ' +str(const_val_left)
			init_pred_left = (col_val,col_time,constraint_left)
			# init_pred_interval_left = find_Interval(init_pred_left)
			error_left,extra_info = find_error_for_predicate(init_pred_left, bucket_number)
			store_error[const_val_left] = error_left
			gain_left = curr_error - error_left
			store_m_best_predicates(gain_left,init_pred_left,bucket_number,extra_info)
		
		
		#finding error for right value
		if const_val_right in store_error:
			error_right = store_error[const_val_right]
			gain_right = curr_error - error_right
		else:
			constraint_right =i + ' ' + op + ' ' + str(const_val_right)
			init_pred_right = (col_val,col_time,constraint_right)
			# init_pred_interval_right = find_Interval(init_pred_right)
			error_right,extra_info= find_error_for_predicate(init_pred_right, bucket_number)
			store_error[const_val_right] = error_right
			gain_right = curr_error - error_right
			store_m_best_predicates(gain_right,init_pred_right,bucket_number,extra_info)
		
		print 'gain = ',gain,'gain_left ',gain_left,'gain_right = ',gain_right
		

		if(gain_left > gain and gain_left >= gain_right):
			const_val = const_val_left
			gain = gain_left
			# init_pred_interval = init_pred_interval_left
		elif(gain_right > gain and gain_right >=gain_left):
			const_val = const_val_right
			gain = gain_right
			# init_pred_interval = init_pred_interval_right
		elif(gain > gain_right and gain > gain_left):
			#if new solution is not good accept it with some probability
			if(gain_left >= gain_right):
				if( displacement > random.random()):
					const_val = const_val_left
					gain = gain_left
					# init_pred_interval = init_pred_interval_left
			else:
				if( displacement > random.random()):
					const_val = const_val_right
					gain = gain_right
					# init_pred_interval = init_pred_interval_right




		#decrease temperature
		T-=1

	constraint =i + ' ' + op + ' ' + str(const_val)
	init_pred = (col_val,col_time,constraint)
	
	return gain,init_pred



def sa():
	#local variables 
	max_gain = 0.0
	best_predicate = (-1,-1,'val >= 0')
	# best_predicate_interval = []
	any_variable_to_add = 0
	best_bucket_number = -1
	curr_error = find_error_for_predicate_target (target_interval)
	#for every signal variable check best predicate that could be generated.
	for signal_variable_name in processed_signal_variable:
		# print 'value of i is',if
		if(processed_signal_variable[signal_variable_name] == 0):
			#here we are calling function that returns us best predicate for i th signal variable for every operator in operator list
			gain = -1
			for op in operator_list:
				for bucket_number in range(n+1):
					gain_op,predicate_op = generate_predicate(signal_variable_name,op, bucket_number , curr_error)
				if(gain_op > gain):
					gain,predicate,pred_bucket_number = gain_op,predicate_op,bucket_number

			any_variable_to_add = 1


			if(gain <= 0):
				print 'got gain 0 for signal variable ',signal_variable_name
				processed_signal_variable[signal_variable_name] = -1
			else:
				print 'For signal variable ',signal_variable_name,'simulated annealing has selected predicate',predicate
				print 'at position ',pred_bucket_number,'getting a gain of',gain
				if(gain > max_gain):
					max_gain = gain
					best_predicate = predicate
					best_bucket_number = pred_bucket_number

	print 'best of the choosen '
	print 'maximum gain',max_gain
	print 'best predicate',best_predicate
	print 'best_bucket_number',best_bucket_number


#main function hard coded the target signal value
if __name__ == "__main__":
		arguments = sys.argv
		print 'name of config file ',arguments[1]
		
		# list of signal variables [signal_variable_name , column_time , column_value]
		global signal_list
		signal_list = []

		# storing minimum and maximum of signal varibale in form signal_variable_name , minimum value , maximum value
		global minmax_of_signal_variables
		minmax_of_signal_variables = []
		
		# list of buckets of the intervals [bucket number,list of interval]
		global influence_list
		influence_list = []

		

		# list of buckets of the intervals [bucket number,list of interval] where every interval is intersected with previous
		# with resolution [0:k]
		global intersected_influence_list
		intersected_influence_list = {}

		
		# arguments to read from config csv_filename,m,k,Tmax,Tmin,trace_length,target_interval
		other_arguments = []
		
		global target_interval
		target_interval = []

		global target_false_interval
		target_false_interval = []

		global pseudo_targets
		pseudo_targets={}

		global pseudo_complement_targets
		pseudo_complement_targets={}

		global pseudo_targets_error
		pseudo_targets_error={}

		global pseudo_complement_targets_error
		pseudo_complement_targets_error = {}

		global m_best_predicates
		m_best_predicates = []

		
		# list of operators to take into consideration while generating predicates
		global operator_list
		operator_list = ['>=','<=']

		#find interval arguments are column number of value,column number of time,predicate
		if(len(arguments) < 2):
			print 'config filename not given in argument so by default taking config_file_constrained.txt'
			config_file_name = 'config_file_constrained.txt'
		else:
			config_file_name = arguments[1]
		print 'calling read_config'
		read_from_config(config_file_name,signal_list,other_arguments,influence_list)
		# print 'influence list \n',influence_list
		# print 'other arguments ',other_arguments
		# target_time = 19
		# target_col = 20
		# pred = 'value >= 2'
		# addedvariable[target_col/2] = 1
		# m=5
		global csv_filemame
		csv_filename = other_arguments[0]
		global m 
		m = int(other_arguments[1])
		k= float(other_arguments[2])
		global Tmax
		global Tmin
		Tmax = int(other_arguments[3])
		Tmin = int(other_arguments[4])
		global n
		n= int(other_arguments[5])
		global trace_length
		trace_length = float(other_arguments[6])
		global initial_trace_length
		initial_trace_length = trace_length
		target_interval = other_arguments[7]
		print 'target_interval \n',target_interval,'\n'

		target_false_interval = complement_interval(target_interval)

		print 'target_false_interval \n',target_false_interval,'\n'

		sort_influence_list(influence_list)
		
		global initial_minimum_bucket_value
		initial_minimum_bucket_value = get_min_bucket_value(influence_list)
		print 'initial_minimum_bucket_value',initial_minimum_bucket_value

		generate_pseudo_targets(pseudo_targets,n,k)
		generate_pseudo_complement_targets(pseudo_complement_targets,n,k)
		
		print 'pseudo targets are as follows'
		print_dictionary(pseudo_targets)

		print 'pseudo_complement_targets'
		print_dictionary(pseudo_complement_targets)


		fill_intersected_influence_list(n)

		global processed_signal_variable
		processed_signal_variable = {}
		for i in signal_list:
			processed_signal_variable[i[0]] = 0
		
		

		# the function computes minimum and maximum value of every signal variable and stores it in list minmax of signla variable
		compute_min_max_of_signal_variable(csv_filename,signal_list ,minmax_of_signal_variables)

		#the signal variable whose min and max values are same will not contribute and hence are eliminated
		eleminate_useless_signals(processed_signal_variable , minmax_of_signal_variables)

		sa()

		print_m_best_predicates()
