# Toxicity Detection in Memes | EVA Application

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/georgia-tech-db/eva-application-template/blob/main/car_plate_detection.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" /> Run on Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/georgia-tech-db/eva-application-template/blob/main/car_plate_detection.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source on GitHub</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/georgia-tech-db/eva-application-template/blob/main/car_plate_detection.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" /> Download notebook</a>
  </td>
</table>
<br>
<br>



### Install Application Dependecies 


```python
pip -q install -r requirements.txt
```

    Note: you may need to restart the kernel to use updated packages.


### Start EVA server

We are reusing the start server notebook for launching the EVA server.


```python
!wget -nc "https://raw.githubusercontent.com/georgia-tech-db/eva/master/tutorials/00-start-eva-server.ipynb"
%run 00-start-eva-server.ipynb
cursor = connect_to_server()
```

    File ‘00-start-eva-server.ipynb’ already there; not retrieving.
    
    nohup eva_server > eva.log 2>&1 &
    
    Note: you may need to restart the kernel to use updated packages.


### Load the Memes for analysis


```python
cursor.execute('DROP TABLE IF EXISTS MemeImages;')
response = cursor.fetch_all()
print(response)
cursor.execute('LOAD IMAGE "meme1.jpg" INTO MemeImages;')
response = cursor.fetch_all()
print(response)
cursor.execute('LOAD IMAGE "meme2.jpg" INTO MemeImages;')
response = cursor.fetch_all()
print(response)
```

    @status: ResponseStatus.SUCCESS
    @batch: 
                                             0
    0  Table Successfully dropped: MemeImages
    @query_time: 0.030441742157563567
    @status: ResponseStatus.SUCCESS
    @batch: 
                                0
    0  Number of loaded IMAGE: 1
    @query_time: 0.055090541020035744
    @status: ResponseStatus.SUCCESS
    @batch: 
                                0
    0  Number of loaded IMAGE: 1
    @query_time: 0.01767381909303367


### Create OCR Extractor UDF


```python
cursor.execute("DROP UDF OCRExtractor;")
response = cursor.fetch_all()
print(response)
cursor.execute("""CREATE UDF IF NOT EXISTS OCRExtractor
      INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
      OUTPUT (labels NDARRAY STR(ANYDIM), bboxes NDARRAY FLOAT32(ANYDIM, 4),
              scores NDARRAY FLOAT32(ANYDIM))
      TYPE  Classification
      IMPL  'ocr_extractor.py';
      """)
response = cursor.fetch_all()
print(response)
```

    @status: ResponseStatus.SUCCESS
    @batch: 
                                            0
    0  UDF OCRExtractor successfully dropped
    @query_time: 0.01484653796069324
    @status: ResponseStatus.SUCCESS
    @batch: 
                                                           0
    0  UDF OCRExtractor successfully added to the database.
    @query_time: 2.548301833216101


### Create Custom UDF for Toxicity Classification


```python
cursor.execute("DROP UDF IF EXISTS ToxicityClassifier;")
response = cursor.fetch_all()
print(response)
cursor.execute("""CREATE UDF IF NOT EXISTS ToxicityClassifier
                  INPUT  (text NDARRAY STR(100))
                  OUTPUT (labels NDARRAY STR(10))
                  TYPE  Classification
                  IMPL  'toxicity_classifier.py';
      """) 
response = cursor.fetch_all()
print(response)
```

    @status: ResponseStatus.SUCCESS
    @batch: 
                                                  0
    0  UDF ToxicityClassifier successfully dropped
    @query_time: 0.011066884966567159
    @status: ResponseStatus.SUCCESS
    @batch: 
                                                                 0
    0  UDF ToxicityClassifier successfully added to the database.
    @query_time: 1.4505423428490758


### Run Toxicity Classifier on OCR Extracted from Images


```python
cursor.execute("""SELECT OCRExtractor(data).labels,
                  ToxicityClassifier(OCRExtractor(data).labels)
                  FROM MemeImages;""")
response = cursor.fetch_all()
print(response)
```

    @status: ResponseStatus.SUCCESS
    @batch: 
                                  ocrextractor.labels toxicityclassifier.labels
    0                  [CANT FuCK WITh, MEIN SWAG E]                     toxic
    1  [YOU CANT SPELL, CLINTON WITHOUT CNN, igfip:]                 not toxic
    @query_time: 6.558598998002708


### Visualize Model Output on Images


```python
from pprint import pprint
from matplotlib import pyplot as plt
import cv2
import numpy as np

def annotate_image(detections, input_image_path, image_id):

    print(detections)

    color1=(207, 248, 64)
    color2=(255, 49, 49)
    thickness=4

    df = detections
    df = df[df.index == image_id]

    image = cv2.imread(input_image_path)

    if df.size:
        df_value_list = df.values
        ocr = ' '.join(df_value_list[0][0])
        label = df_value_list[0][1]
        print(label)

        plt.imshow(image)
        plt.show()

        cv2.putText(image, label, (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, color2, thickness, cv2.LINE_AA) 

        cv2.putText(image, ocr, (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color1, thickness, cv2.LINE_AA) 

        plt.imshow(image)
        plt.show()        
```


```python
from ipywidgets import Image
dataframe = response.batch.frames
annotate_image(dataframe, 'meme1.jpg', image_id=0)
annotate_image(dataframe, 'meme2.jpg', image_id=1)
```

                                 ocrextractor.labels toxicityclassifier.labels
    0                  [CANT FuCK WITh, MEIN SWAG E]                     toxic
    1  [YOU CANT SPELL, CLINTON WITHOUT CNN, igfip:]                 not toxic
    toxic



    
![png](README_files/README_16_1.png)
    



    
![png](README_files/README_16_2.png)
    


                                 ocrextractor.labels toxicityclassifier.labels
    0                  [CANT FuCK WITh, MEIN SWAG E]                     toxic
    1  [YOU CANT SPELL, CLINTON WITHOUT CNN, igfip:]                 not toxic
    not toxic



    
![png](README_files/README_16_4.png)
    



    
![png](README_files/README_16_5.png)
    

