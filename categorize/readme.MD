

###Categorize Results


<table>
  <tr>
    <th>Samples</th>
    <th>Knearest</th>        
    <th>SVM</th>        
    <th>RTrees</th>        
    <th>Boost</th>        
    <th>MLP</th>
    <th>Size</th>
    <th>Options</th>
  <tr>
    <td>997</td>
    <td>30</td>
    <td>24</td>
    <td>32</td>
    <td>29</td>
    <td>26</td>
    <td>100</td>
    <td>None</td>
  </tr>
  <tr>
    <td>997</td>
    <td>28</td>
    <td>23</td>
    <td>26</td>
    <td>36</td>
    <td>24</td>
    <td>100</td>
    <td>Autocrop</td>
  </tr>
  <tr>
    <td>2988</td>
    <td>22.74</td>
    <td>22.07</td>
    <td>26.08</td>
    <td>27.75</td>
    <td>24.41</td>
    <td>100</td>
    <td>None</td>
  </tr>
  <tr>
    <td>2988</td>
    <td>19.73</td>
    <td>19.06</td>
    <td>23.74</td>
    <td>27.09</td>
    <td>20.06</td>
    <td>100</td>
    <td>Autocrop</td>
  </tr>
  <tr>
    <td>3985</td>
    <td>22.30</td>
    <td>24.31</td>
    <td>26.81</td>
    <td>27.56</td>
    <td>27.81</td>
    <td>100</td>
    <td>None</td>
  </tr>  
  <tr>
    <td>3985</td>
    <td>23.30</td>
    <td>22.55</td>
    <td>28.07</td>
    <td>29.82</td>
    <td>25.31</td>
    <td>100</td>
    <td>Autocrop</td>
  </tr>
  <tr>
    <td>4984</td>
    <td>21.64</td>
    <td>21.64</td>
    <td>28.25</td>
    <td>29.65</td>
    <td>22.44</td>
    <td>100</td>
    <td>None</td>
  </tr> 
  <tr>
    <td>4984</td>
    <td>19.63</td>
    <td>20.04</td>
    <td>28.25</td>
    <td>31.86</td>
    <td>20.84</td>
    <td>100</td>
    <td>Autocrop</td>
  </tr> 
  <tr>
    <td>5980</td>
    <td>22.90</td>
    <td>22.90</td>
    <td>27.42</td>
    <td>28.42</td>
    <td>24.08</td>
    <td>100</td>
    <td>None</td>
  </tr>
  <tr>
    <td>5980</td>
    <td>19.23</td>
    <td>19.89</td>
    <td>24.91</td>
    <td>28.26</td>
    <td>21.23</td>
    <td>100</td>
    <td>Autocrop</td>
  </tr>
</table>


Samples,Knearest,SVM,RTrees,Boost,MLP
2923,22.53,21.50,25.26,27.99,22.18
2970,24.24,22.90,24.24,31.99,22.90
2987,22.41,21.74,24.75,25.08,22.07
2988,22.74,22.07,26.08,27.75,24.41
3036,22.04,20.72,24.01,30.59,21.05
3207,22.74,21.50,25.23,28.04,24.61
3298,21.82,20.61,24.85,30.00,24.85
4430,17.38,22.12,25,73,27.99,21.67
4672,20.08,23.29,27.56,30.98,25.21
6608,20.57,21.63,26.17,28.59,22.99


return cv2.adaptiveThreshold(cv2.resize(cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE), (SZ,SZ)),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
AdaptiveThreshold thing
0997,30.0,30.0,33.0,36.0,34.0
3985,25.5639097744,23.3082706767,28.320802005,29.0726817043,23.0576441103


Auto crop
6608,18.30,16.94,24.05,27.83,16.79
