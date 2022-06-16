# importing the necessary libraries
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
from obspy import read
from datetime import datetime
from statistics import mean, stdev

# paths
MPpath = './mseed/mp'
VTBpath = './mseed/vtb'

# defining a function for listing available files in a path/folder
def openFolder(path):
    listOfFileNames = []
    for file in os.listdir(path):
        listOfFileNames.append(file)
    return listOfFileNames

# assigning the files list to a variable
MPlist = openFolder(MPpath)
VTBlist = openFolder(VTBpath)

# turning of interactive plot to enable matplotlib.pyplot.savefig
plt.ioff()

# define triggering algorithm
def triggerFunc(trace_array, charFunc, thres, method, saveFilePath):
    
    # define empty array for the detected event
    list = []
    
    # iterate for every element in the character function
    for i in range(len(charFunc)):
        # if an element is greater than equal to the threshold
        if charFunc[i] >= thres:
            # append the element to the empty array
            list.append(charFunc[i])
            # break the for loop
            break
    
    # getting the indice of the detected event element
    if len(list) > 0:
      event_indice = charFunc.index(list[0])
    else:
      event_indice = 0
    
    # plotting the trigger
    station = trace_array.stats.station
    loc = trace_array.stats.location
    chn = trace_array.stats.channel

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(trace_array.times('timestamp'), charFunc)
    ax1.axvline(trace_array.times('timestamp')[event_indice], color='red')
    ax1.set_title(method + ' characteristic function of ' + station + '.' + loc + '.' + chn)

    ax2.plot(trace_array.times('timestamp'), trace_array.data)
    ax2.axvline(trace_array.times('timestamp')[event_indice], color='red')
    ax2.set_title(method + ' applied to seismogram of ' + station + '.' + loc + '.' + chn)

    fig.tight_layout()
    fig.savefig('./results/' + saveFilePath + '.png', dpi=300)

    utc = datetime.utcfromtimestamp(trace_array.times('timestamp')[event_indice]).strftime('%Y-%m-%dT%H:%M:%S')

# defining the STA/LTA-A function
def sta_lta(a, nsta_s, nlta_s):

    '''
    a       --> trace array
    nsta_s  --> STA window size in seconds
    nlta_s  --> LTA window size in seconds
    delta_t --> trace sample distance in seconds
    nsta    --> STA window in samples
    nlta    --> LTA window in samples
    '''

    # finding the number of samples in a window from the window length
    delta_t = a.stats.delta
    nsta = int((nsta_s / delta_t) + 1)
    nlta = int((nlta_s / delta_t) + 1)

    # finding the coefficients
    csta = 1. / nsta # coefficient for STA
    clta = 1. / nlta # coefficient for LTA

    # define empty arrays
    staList = []
    ltaList = []
    chrFunc = []

    # looping the whole STA/LTA window
    for i in range(len(a) - (nsta + nlta) + 1):    
      sta_n = 0
      lta_n = 0

      # looping the STA window
      for j in range(len(a[i:i + nsta])):
          sta_n += a[i + j]
      STA = csta * sta_n
      staList.append(STA)

      # looping the LTA window
      for k in range(len(a[i:i + nlta])):
          lta_n += a[i + k + nsta]
      LTA = clta * lta_n
      ltaList.append(LTA)
    
    # acquiring the STA/LTA ratio
    for i in range(len(staList)):
      ratio = staList[i] / ltaList[i]
      chrFunc.append(ratio)

    # filling the gap
    minima = min(chrFunc)
    minima_sta = min(staList)
    minima_lta = min(ltaList)
    for i in range(len(a) - len(chrFunc)):
      chrFunc.append(minima)
      staList.append(minima_sta)
      ltaList.append(minima_lta)
  
    return [chrFunc, staList, ltaList]

# defining the STA/LTA-B function
def sta_lta_abs(a, nsta_abs_s, nlta_abs_s):

    '''
    a           --> trace array
    nsta_abs_s  --> STA window size in seconds
    nlta_abs_s  --> LTA window size in seconds
    delta_t     --> trace sample distance in seconds
    nsta_abs    --> STA window in samples
    nlta_abs    --> LTA window in samples
    '''

    # finding the number of samples in a window from the window length
    delta_t = a.stats.delta
    nsta_abs = int((nsta_abs_s / delta_t) + 1)
    nlta_abs = int((nlta_abs_s / delta_t) + 1)

    # finding the coefficients
    csta = 1. / nsta_abs # coefficient for STA
    clta = 1. / nlta_abs # coefficient for LTA

    # define empty arrays
    staList = []
    ltaList = []
    chrFunc = []

    # looping the whole STA/LTA window
    for i in range(len(a) - (nsta_abs + nlta_abs) + 1):    
      sta_n = 0
      lta_n = 0

      # looping the STA window
      for j in range(len(a[i:i + nsta_abs])):
          sta_n += abs(a[i + j])
      STA = csta * sta_n
      staList.append(STA)

      # looping the LTA window
      for k in range(len(a[i:i + nlta_abs])):
          lta_n += abs(a[i + k + nsta_abs])
      LTA = clta * lta_n
      ltaList.append(LTA)
    
    # acquiring the STA/LTA ratio
    for i in range(len(staList)):
      ratio = staList[i] / ltaList[i]
      chrFunc.append(ratio)

    # filling the gap
    minima = min(chrFunc)
    minima_sta = min(staList)
    minima_lta = min(ltaList)
    for i in range(len(a) - len(chrFunc)):
      chrFunc.append(minima)
      staList.append(minima_sta)
      ltaList.append(minima_lta)
  
    return [chrFunc, staList, ltaList]

# defining the LTE/STE function
def lte_ste (a, nlte, nste):
    clte = 1. / nlte # coefficient for LTE
    cste = 1. / nste # coefficient for STE
    
    # define empty arrays
    chrFunc = []
    lteList = []
    steList = []
    
    for i in range(len(a)):
        
        # define initial values
        numste = 0
        numlte = 0
        ste = 0
        lte = 0
        
        # LTE numerator
        for j in range((i - nlte), i):
            numlte += (a[j]**2)
        
        # STE numerator
        for k in range((i - nste), i):
            numste += (a[k]**2)
        
        # LTE and STE
        lte = clte * numlte
        ste = cste * numste
        lteList.append(lte)
        steList.append(ste)
        
        # LTE/STE
        chrFunc.append(lte / ste)
    
    # returns an array of LTE/STE ratios
    return [chrFunc, lteList, steList]

# defining the MER function
def mer(a, L_s):

    '''
    a           --> trace array
    L_s         --> ER window size in seconds
    L           --> ER window size in samples
    delta_t     --> trace sample distance in seconds
    '''

    # finding the number of samples in a window from the window length
    delta_t = a.stats.delta
    L = int((L_s / delta_t) + 1)

    # define empty arrays
    leftMERlist = []
    rightMERlist = []
    chrFunc = []

    # looping the whole ER window
    for i in range(len(a) - (2 * L) + 1):    
      leftMER_n = 0
      rightMER_n = 0

      # looping the left ER window
      for j in range(len(a[i:i + L])):
          leftMER_n += (a[i + j])**2
      leftMERlist.append(leftMER_n)

      # looping the right ER window
      for k in range(len(a[i:i + L])):
          rightMER_n += (a[i + k + L])**2
      rightMERlist.append(rightMER_n)
    
    # acquiring the characteristic function
    for i in range(len(leftMERlist)):
      mer = ((rightMERlist[i] / leftMERlist[i]) * abs(a[i]))**3
      chrFunc.append(mer)

    # filling the gap
    minima = min(chrFunc)
    minima_right = min(rightMERlist)
    minima_left = min(leftMERlist)
    for i in range(len(a) - len(chrFunc)):
      chrFunc.append(minima)
      rightMERlist.append(minima_right)
      leftMERlist.append(minima_left)

    return [chrFunc, rightMERlist, leftMERlist]

# defining the Kurtosis function
def kurtosis (a, T):

  # finding the number of samples K in a window from the window length T
  delta_t = a.stats.delta
  K = int((T / delta_t) + 1)

  '''
  a --> trace array
  T --> window length in seconds
  K --> number of samples in a window
  '''

  # acquiring the mean and standard deviation from the trace array
  a_mean = mean(a)
  a_std = stdev(a)
   
  # define an empty array
  kurtosis_array = []
    
  # define initial values
  coef = 1. / ((K - 1) * (a_std**4))
    
  # starting the loop
  for i in range(len(a) - K + 1):

    # resetting the value
    avg = 0

    # looping between values in the window
    for j in range(len(a[i:i + K])):
      avg += ((a[i+1] - a_mean)**4)
      
    # using the final value from the previous loop to calculate the kurtosis
    kurtosis = (-3) + (coef * avg)

    # appending the kurtosis value in the empty array
    kurtosis_array.append(kurtosis)

  # minimum value in the current state of kurtosis_array
  minima = min(kurtosis_array)

  # adding the minimum value
  for k in range(len(a) - len(kurtosis_array)):
    kurtosis_array.append(minima)
  
  # return a 2D array of arrays and values
  return [kurtosis_array, coef]

# execution
print(
  '''
  -------------------------------------------------------
  copyright@rifqi_rahmandito
  kindly leave a comment to rifqi_rahmandito@yahoo.com :)
  -------------------------------------------------------
  '''
)

dir = str(input(
  '''
  ------------------------------------
  Input master directory (mp or vtb): 
  ------------------------------------
  --> '''
))

if dir == 'mp':
  print('  There are %r filenames in the directory' %len(MPlist))
  method = str(input(
    '''
    -------------------------------------------------
    Input desired automation method:
    (sta-lta / sta-lta-abs / lte-ste / mer / kurtosis)
    -------------------------------------------------
    --> '''
  ))

  # STA/LTA
  if method == 'sta-lta':
    for i in range(len(MPlist)):
      a = read('./mseed/mp/' + MPlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a, sta_lta(a, 0.009, 4.5)[0], 115.0, str(i+1) + '-STA/LTA', 'mp/sta-lta/MP-STA-LTA-' + str(i+1))

  # STA/LTA ABS
  elif method == 'sta-lta-abs':
    for i in range(len(MPlist)):
      a = read('./mseed/mp/' + MPlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a, sta_lta_abs(a, 0.01, 3.0)[0], 0.2, str(i+1) + '-STA/LTA-ABS', 'mp/sta-lta-abs/MP-STA-LTA-ABS-' + str(i+1))

  # LTE/STE
  elif method == 'lte-ste':
    for i in range(len(MPlist)):
      a = read('./mseed/mp/' + MPlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a, lte_ste(a, int(9.0*sRate), int(2.5*sRate))[0], np.max(lte_ste(a, int(9.0*sRate), int(2.5*sRate))), str(i+1) + '-LTE/STE', 'mp/lte-ste/MP-LTE-STE-' + str(i+1))
  
  # MER
  elif method == 'mer':
    for i in range(len(MPlist)):
      a = read('./mseed/mp/' + MPlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a.taper(0.5, side='left'), mer(a, 3.05)[0], 150.0, str(i+1) + '-MER', 'mp/mer/MP-MER-' + str(i+1))
  
  # Kurtosis
  elif method == 'kurtosis':
    for i in range(len(MPlist)):
      a = read('./mseed/mp/' + MPlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a, kurtosis(a, 1.75)[0], 50.0, str(i+1) + '-Kurtosis', 'mp/kurtosis/MP-KURTOSIS-' + str(i+1))
  
  else:
    print('error: no such option')

elif dir == 'vtb':
  print('  There are %r filenames in the directory' %len(VTBlist))
  method = str(input(
    '''
    -------------------------------------------------
    Input desired automation method:
    (sta-lta / sta-lta-abs / lte-ste / mer / kurtosis)
    -------------------------------------------------
    --> '''
  ))

  # STA/LTA
  if method == 'sta-lta':
    for i in range(len(VTBlist)):
      a = read('./mseed/vtb/' + VTBlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a, sta_lta(a, 0.009, 4.5)[0], 115.0, str(i+1) + '-STA/LTA', 'vtb/sta-lta/VTB-STA-LTA-' + str(i+1))

  # STA/LTA ABS
  elif method == 'sta-lta-abs':
    for i in range(len(VTBlist)):
      a = read('./mseed/vtb/' + VTBlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a, sta_lta_abs(a, 0.01, 3.0)[0], 0.2, str(i+1) + '-STA/LTA-ABS', 'vtb/sta-lta-abs/VTB-STA-LTA-ABS-' + str(i+1))

  # LTE/STE
  elif method == 'lte-ste':
    for i in range(len(VTBlist)):
      a = read('./mseed/vtb/' + VTBlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a, lte_ste(a, int(9.0*sRate), int(2.5*sRate))[0], np.max(lte_ste(a, int(9.0*sRate), int(2.5*sRate))), str(i+1) + '-LTE/STE', 'vtb/lte-ste/VTB-LTE-STE-' + str(i+1))
  
  # MER
  elif method == 'mer':
    for i in range(len(VTBlist)):
      a = read('./mseed/vtb/' + VTBlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a.taper(0.5, side='left'), mer(a, 3.05)[0], 1.0, str(i+1) + '-MER', 'vtb/mer/VTB-MER-' + str(i+1))
  
  # Kurtosis
  elif method == 'kurtosis':
    for i in range(len(VTBlist)):
      a = read('./mseed/vtb/' + VTBlist[i])[0]
      sRate = a.stats.sampling_rate
      triggerFunc(a, kurtosis(a, 1.75)[0], -2.9, str(i+1) + '-Kurtosis', 'vtb/kurtosis/VTB-KURTOSIS-' + str(i+1))
  
  else:
    print('error: no such options')

else:
  print('error: no such option')