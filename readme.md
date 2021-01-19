# Udacity Enron Fraud Detection Machine Learning Project

## Machine learning project using python 2.7 and sklearn to find out the best way to detect fraud from financial data and a set of emails taken from Enron employees. The below is the output of the program.

___

 - total number of data points:  146

 - allocation across classes (POI/non-POI):  18 / 128

 - number of features used:  21

 are there features with many missing values? etc. Yes, and here
they are

 - Number people no salary:  51

 - Number NaN payments:  21

 - percent NaN payments:  0.143835616438

 - Number of POIs no payments:  0

 - percent poi NaN payments:  0.0

 - Sample person from dataset:  METTS MARK

 - Features:  {'salary': 365788, 'to_messages': 807, 'deferral_pa
yments': 'NaN', 'total_payments': 1061827, 'exercised_stock_optio
ns': 'NaN', 'bonus': 600000, 'restricted_stock': 585062, 'shared_
receipt_with_poi': 702, 'restricted_stock_deferred': 'NaN', 'tota
l_stock_value': 585062, 'expenses': 94299, 'loan_advances': 'NaN'
, 'from_messages': 29, 'other': 1740, 'from_this_person_to_poi':
1, 'poi': False, 'director_fees': 'NaN', 'deferred_income': 'NaN'
, 'long_term_incentive': 'NaN', 'email_address': 'mark.metts@enro
n.com', 'from_poi_to_this_person': 38}

 - Total number of POIs:
n    31
y     4

 - Total number of POIs where we have their emails:  4

   POI                  Name
0    y          Lay, Kenneth
24   y       Delainey, David
19   y          Forney, John
1    y     Skilling, Jeffrey
31   n        Bowen, Raymond
30   n         Duncan, David
29   n       Kopper, Michael
28   n       Belden, Timothy
27   n         Lawyer, Larry
18   n          Koenig, Mark
25   n           Glisan, Ben
32   n       Colwell, Wesley
23   n        Fastow, Andrew
22   n           Fastow, Lea
21   n         Rieker, Paula
20   n         Rice, Kenneth
33   n            Boyle, Dan
26   n      Richter, Jeffrey
17   n         Hannon, Kevin
16   n      DeSpain, Timothy
15   n   Calger, Christopher
14   n       Causey, Richard
13   n         Fuhs, William
12   n         Furst, Robert
11   n          Brown, James
10   n        Bayley, Daniel
9    n         Mulgrew, Gary
8    n          Darby, Giles
7    n     Bermingham, David
6    n           Shelby, Rex
5    n         Hirko, Joseph
4    n         Yeager, Scott
3    n       Krautz, Michael
2    n         Howard, Kevin
34   n    Loehr, Christopher


10% or more emails from POI:  UMANOFF ADAM S 0.108108108108
10% or more emails from POI:  COLWELL WESLEY 0.136518771331
10% or more emails from POI:  DEFFNER JOSEPH M 0.16106442577
10% or more emails from POI:  KISHKILL JOSEPH G 0.16106442577
10% or more emails from POI:  REDMOND BRIAN L 0.122082585278
10% or more emails from POI:  BAZELIDES PHILIP J 0.122082585278
10% or more emails from POI:  DURAN WILLIAM D 0.117256637168
10% or more emails from POI:  THORN TERENCE H 0.117256637168
10% or more emails from POI:  FASTOW ANDREW S 0.117256637168
10% or more emails from POI:  FOY JOE 0.117256637168
10% or more emails from POI:  DIETRICH JANET R 0.118584758942
10% or more emails from POI:  DONAHUE JR JEFFREY M 0.217341040462


What is the accuracy score?  0.8409090909090909

How many POIs are predicted for the test set?  5.0

How many people total are in your test set? 44

If your identifier predicted 0. (not POI), what would its accurac
y be? 0.8863636363636364

Do you get any true positives?  ['yes']

poi precision:  0.2

poi recall:  0.25


 minimum stock options exercised:  -44093

 maximum stock options exercised:  49110078

 minimum bonus:  70000

 max bonus:  8000000

 minimum email to poi:  0.0093023255814

 max email to poi:  1.0
Rescaled $8,000,000 bonus and $1,000,000 exercised stock options:
  [[0.00875    0.16365026]]

 accuracy:  0.8863636363636364
Do you get any true positives? (new)  ['yes']

poi precision: (new) 0.2

poi recall: (last print) 0.25
