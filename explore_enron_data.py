#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#print(len(length_dict))

#print(len(enron_data["SKILLING JEFFREY K"]))

names={key for key, value in enron_data.items()}
print(sum(1 for name in names if enron_data[name]["poi"]==1))
# easier way in one line is below

'''
print "total no. of poi=",sum(1 for key in enron_data if enron_data[key]["poi"]==1)

print "total_stock_value of PRENTICE, JAMES =",enron_data["PRENTICE JAMES"]["total_stock_value"]

print "no. of emails from COLWELL, WESLEY to poi =",enron_data["COLWELL WESLEY"]['from_this_person_to_poi']

print "exercised_stock_options of SKILLING, JEFFREY K to poi =",enron_data["SKILLING JEFFREY K"]['exercised_stock_options']

print "total payments for SKILLING JEFFREY K =",enron_data["SKILLING JEFFREY K"]['total_payments']
print "total payments for LAY KENNETH=",enron_data["LAY KENNETH L"]['total_payments']
print "total payments for FASTOW ANDREW=",enron_data["FASTOW ANDREW S"]['total_payments']
'''
print(sum(1 for name in names if enron_data[name]["salary"]!="NaN"))
print(sum(1 for name in names if enron_data[name]["email_address"]!="NaN"))

no_pmt=sum(1 for name in names if enron_data[name]["total_payments"]=="NaN")
print no_pmt
print no_pmt*100/len(enron_data),"%"

no_pmt_poi=sum(1 for name in names if (enron_data[name]["poi"]==1 and enron_data[name]["total_payments"]=="NaN"))
print no_pmt_poi
print no_pmt_poi*100/(sum(1 for name in names if enron_data[name]["poi"]==1)),"%"

print no_pmt
print(len(enron_data))




