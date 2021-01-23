# Udacity Enron Fraud Detection Machine Learning Project

## Machine learning project using python 2.7 and sklearn to find out the best way to detect fraud from financial data and a set of emails taken from Enron employees. The below is the output of the program.

___

<p>The goal for this project is to try to predict who are persons of interest from how frequently someone corresponded via email with a known person of interest and by comparing various financial data points like the amound of the employee's salary, bonus, total stock value, and stock options exercised. The machine learning library sklearn is used for predictions and to process the data and present visualizations using matplotlib. One major outlier in the data is that the total for all financial data points is included as a line in the data. This abviously makes it look like there was possibly one person who made enough money it couldn't possibly be legal. Once that record is removed a clearer picture starts to emerge. </p>

<p>Some output from the file is here on the readme, but the final version of the project is the jupyter notebook poi_id.ipynb although there's also a python 3 version .py file made from that notebook to in the py3versionFinal folder.</p>

<p>Total number of data points:  146</li>
<p>Allocation across classes (POI/non-POI):  18 / 128</li>
<p>Number of features used:  21</li>
<p>Are there features with many missing values? etc. Yes, and here they are:</li>
<ul>
 <li>Number people with no salary data:  51</li>
 <li>Number NaN payments:  21</li>   
 <li>percent NaN payments:  0.143835616438</li>
 <li>Number of POIs no payments:  0</li>
 <li>percent poi NaN payments:  0.0</li>
</ul>

<p>The features I found that best aligned with more accurately predicting persons of interest were the financial points bonus and stock options exercised. I did explore whether or not an employee having a high percentage of email correspondence with a known person of interest correlated with them being a known person of interest, higher bonus or salary, or higher stock values. My hypothesis was that there would be a increased amount of cummunications among those who committed insider trading that correlated to them having a higher bonus or stock option value. There was a lot of noise on the scatter plot for this though and no clear regression pattern to follow, although most persons of interest did seem to fall around the 25% mark for communication with a POI. </p>

<p>I used MinMaxScaler to adjust the scale for the financial features to be comparable to email frequency then plotted the data against bonus amounts and used kmeans clustering to show in different shades of blue the groups of possible POIs and non-POIs. This may have failed because we only have email data on four known persons of interest. Or I was just wrong. At any rate, my dreams of Robert Mueller and James Comey coming to me for advice on solving crimes is not a likely scenario. </p>

<p>I adjusted the parameters many times and barely got them high enough by just manually adjusting things. I added pipeline and a classifier then discovered that could've saved me a tin of time trying to find the right parameters, especially for SVC. I don't think I would've even known to check most of them.</p> 

<p>Results of Decision Tree Manually Tuned</p>
<ul>
 <li>What is the accuracy score?  0.813953488372093</li>
 <li>How many POIs are predicted for the test set?  8.0</li>
 <li>How many people total are in your test set? 43</li>
 <li>If your identifier predicted 0. (not POI), what would its accuracy be? 0.8863636363636364</li>
 <li>Do you get any true positives?  ['yes']</li>
 <li>poi precision:  0.5</li>
 <li>poi recall:  0.5</li>
</ul>

<p>I manually tried adjusting the C and gamma values for the SVC. This was tedious and adjusting too far one way or another would make the recall and precision go to 0. </p>

<p>For validation, I printed the accuracy score, the number of POIs predicted, total number of people in the set, hypothetical accuracy if the prediction had been zero, count of the number of true positives, the precision score, and the recall score. The clustered scatter plot shows which people are predicted and which are true positives(marked with a red X)</p>

<ul>
 <li>What is the accuracy score?  0.8409090909090909</li>
 <li>How many POIs are predicted for the test set?  5.0</li>
 <li>How many people total are in your test set? 44</li>
 <li>If your identifier predicted 0. (not POI), what would its accuracy be? 0.8863636363636364</li>
 <li>Do you get any true positives?  ['yes']</li>
 <li>poi precision:  0.2</li>
 <li>poi recall:  0.25</li>
</ul>

___

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

<p>I wanted to find if there was a connection to the amount of correspondence between POIs and possible POIs</p>

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


 - minimum stock options exercised:  -44093

 - maximum stock options exercised:  49110078

 - minimum bonus:  70000

 - max bonus:  8000000

 - minimum email to poi:  0.0093023255814

-  max email to poi:  1.0

- Rescaled $8,000,000 bonus and $1,000,000 exercised stock options:
    [[0.00875    0.16365026]]

Final answer was SVC

Accuracy:  0.8372093023255814
 
True positives  3 

Final precision:  0.5 

Final recall:  0.42857142857142855 

- poi recall for SVC: 0.25
