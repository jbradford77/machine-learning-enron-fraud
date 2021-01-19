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

<table>
 <thead>
  <tr>
   <th>POI</th>
   <th>Name</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>y</td>
   <td>Lay, Kenneth</td>
  </tr>
  <tr>
   <td>y</td>
   <td> Delainey, David</td>
  </tr>
  <tr>
   <td>y</td>
   <td>Forney, John</td>
  </tr>
  <tr>
   <td>y</td>
   <td>Skilling, Jeffrey</td>
  </tr>
  <tr>
   <td>n</td>
   <td>Bowen, Raymond</td>
  </tr>
    <tr>
    <td>n</td>
    <td>Duncan, David</td>
   </tr>
    <tr>
   <td>n</td>
   <td>Kopper, Michael</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Belden, Timothy</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Lawyer, Larry</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Koenig, Mark</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Glisan, Ben</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Colwell, Wesley</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Fastow, Andrew</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Fastow, Lea</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Rieker, Paula</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Rice, Kenneth</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Boyle, Dan</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Richter, Jeffrey</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Hannon, Kevin</td>
  </tr>
    <tr>
   <td>n</td>
   <td>DeSpain, Timothy</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Calger, Christopher</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Causey, Richard</td>
  </tr>
    <tr>
   <td>n</td>
   <td>Fuhs, William</td>
  </tr>
    <tr>
     <td>n</td>
     <td>Brown, James</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Bayley, Daniel</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Mulgrew, Gary</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Darby, Giles</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Bermingham, David</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Shelby, Rex</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Hirko, Joseph</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Yeager, Scott</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Krautz, Michael</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Howard, Kevin</td>
    </tr>
   <tr>
     <td>n</td>
     <td>Loehr, Christopher</td>
    </tr>
 </tbody>
</table>


<p>10% or more emails from POI:  UMANOFF ADAM S     0.108108108108</p>
<p>10% or more emails from POI:  COLWELL WESLEY     0.136518771331</p>
<p>10% or more emails from POI:  DEFFNER JOSEPH M   0.16106442577</p>
<p>10% or more emails from POI:  KISHKILL JOSEPH G  0.16106442577</p>
<p>10% or more emails from POI:  REDMOND BRIAN L    0.122082585278</p>
<p>10% or more emails from POI:  BAZELIDES PHILIP J 0.122082585278</p>
<p>10% or more emails from POI:  DURAN WILLIAM D    0.117256637168</p>
<p>10% or more emails from POI:  THORN TERENCE H    0.117256637168</p>
<p>10% or more emails from POI:  FASTOW ANDREW S    0.117256637168</p>
<p>10% or more emails from POI:  FOY JOE            0.117256637168</p>
<p>10% or more emails from POI:  DIETRICH JANET R   0.118584758942</p>
<p>10% or more emails from POI:  DONAHUE JR JEFFREY 0.217341040462</p>


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
